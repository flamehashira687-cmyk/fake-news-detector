
from pathlib import Path
import re
import sqlite3
from datetime import datetime
import os

import joblib
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fake_news_model.joblib"
DB_PATH = BASE_DIR / "truthlens.db"
FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Load the API key from the local .env file.
load_dotenv(BASE_DIR / ".env")
FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip()

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run python train_model.py first."
    )

model = joblib.load(MODEL_PATH)


# ---------- DATABASE ----------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                news_text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                confidence_level TEXT NOT NULL,
                suspicious_phrases TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def save_analysis(text, result):
    phrases = ", ".join(result["suspicious_phrases"])

    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO analyses (
                news_text, prediction, confidence,
                confidence_level, suspicious_phrases,
                word_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            text,
            result["prediction"],
            result["confidence"],
            result["confidence_level"],
            phrases,
            result["word_count"],
            datetime.now().astimezone().strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            ),
        ))

        return cursor.lastrowid


# ---------- EXPLAINABLE AI ----------

def explain_prediction(text, prediction, top_n=8):
    """
    Explain binary Logistic Regression predictions using
    TF-IDF-weighted feature contributions when supported.
    These are model signals, not evidence of factual truth.
    """

    explanation = {
        "available": False,
        "important_words": [],
        "method": "TF-IDF-weighted Logistic Regression coefficients",
        "message": (
            "Feature-level explanation is unavailable for "
            "this saved model format."
        ),
    }

    # The model must be a compatible sklearn Pipeline.
    if not hasattr(model, "named_steps"):
        return explanation

    steps = model.named_steps

    vectorizer = next(
        (
            step for step in steps.values()
            if hasattr(step, "get_feature_names_out")
            and hasattr(step, "transform")
        ),
        None,
    )

    classifier = next(
        (
            step for step in steps.values()
            if hasattr(step, "coef_")
            and hasattr(step, "classes_")
        ),
        None,
    )

    if vectorizer is None or classifier is None:
        return explanation

    try:
        matrix = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()

        if len(classifier.classes_) != 2:
            explanation["message"] = (
                "Feature explanations currently support "
                "binary classification only."
            )
            return explanation

        classes = list(classifier.classes_)
        predicted_class_index = classes.index(prediction)
        coefficients = classifier.coef_[0]

        # Use non-zero TF-IDF features in this specific article.
        row = matrix.getrow(0)
        present_indices = row.indices
        present_values = row.data

        contributions = []

        for index, tfidf_value in zip(
            present_indices, present_values
        ):
            # For binary Logistic Regression, coef_ supports classes_[1].
            contribution = float(
                coefficients[index] * tfidf_value
            )

            # Reverse direction when the predicted class is classes_[0].
            if predicted_class_index == 0:
                contribution = -contribution

            if contribution == 0:
                continue

            contributions.append({
                "word": str(feature_names[index]),
                "contribution": round(contribution, 4),
                "direction": (
                    "supports prediction"
                    if contribution > 0
                    else "opposes prediction"
                ),
            })

        contributions.sort(
            key=lambda item: abs(item["contribution"]),
            reverse=True,
        )

        explanation["available"] = True
        explanation["important_words"] = contributions[:top_n]
        explanation["message"] = (
            "Words are ranked by the absolute TF-IDF-weighted "
            "linear contribution for this prediction. Positive "
            "values support the predicted class; negative values "
            "oppose it. These are model signals, not proof of truth."
        )

        return explanation

    except (
        ValueError,
        IndexError,
        AttributeError,
        TypeError,
    ):
        return explanation


# ---------- MODEL ANALYSIS ----------

def find_suspicious_phrases(text):
    patterns = [
        r"\bbreaking\b",
        r"\bshocking\b",
        r"\bmust see\b",
        r"\bshare this\b",
        r"\bthey don't want you to know\b",
        r"\b100% guaranteed\b",
        r"\bsecret truth\b",
        r"\bmainstream media\b",
    ]

    found = []

    for pattern in patterns:
        found.extend(
            re.findall(pattern, text, flags=re.IGNORECASE)
        )

    return sorted(set(found), key=str.lower)


def analyze_news(text):
    text = text.strip()

    if len(text) < 30:
        return {
            "error": (
                "Please enter a news article with at least "
                "30 characters."
            )
        }

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    classes = list(model.classes_)
    predicted_index = classes.index(prediction)
    confidence = float(probabilities[predicted_index]) * 100

    if confidence < 60:
        confidence_level = "Low"
    elif confidence < 80:
        confidence_level = "Moderate"
    else:
        confidence_level = "High"

    # Generate feature-level explanation.
    explanation = explain_prediction(text, prediction)

    return {
        "prediction": str(prediction),
        "confidence": round(confidence, 2),
        "confidence_level": confidence_level,
        "suspicious_phrases": find_suspicious_phrases(text),
        "word_count": len(text.split()),
        "explanation": explanation,
        "model_type": "TF-IDF + Logistic Regression",
        "disclaimer": (
            "This is an ML prediction, not proof of factual truth. "
            "Model confidence may not be calibrated. Verify important "
            "claims using independent, reliable sources."
        ),
    }


# ---------- WEBSITE ----------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/analyze", methods=["POST"])
def analyze_form():
    text = request.form.get("news_text", "")
    result = analyze_news(text)

    if "error" not in result:
        save_analysis(text.strip(), result)

    return render_template(
        "index.html",
        news_text=text,
        result=result,
    )


# ---------- ANALYSIS API ----------

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str):
        return jsonify({
            "error": "News text must be a string."
        }), 400

    result = analyze_news(text)

    if "error" in result:
        return jsonify(result), 400

    analysis_id = save_analysis(text.strip(), result)
    result["id"] = analysis_id

    return jsonify(result)


# ---------- SOURCE VERIFICATION ----------

@app.route("/api/verify", methods=["POST"])
def api_verify():
    """
    Search Google's Fact Check Tools API for matching published
    fact-check reviews. A result is not automatically a verdict
    on the article being analyzed.
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str):
        return jsonify({"error": "News text must be a string."}), 400

    text = text.strip()
    if len(text) < 30:
        return jsonify({
            "error": "Please provide at least 30 characters to search."
        }), 400

    if not FACT_CHECK_API_KEY:
        return jsonify({
            "available": False,
            "results": [],
            "message": (
                "Fact-check search is not configured. Add "
                "GOOGLE_FACT_CHECK_API_KEY to your local .env file "
                "and restart the Flask server."
            )
        }), 503

    # Search with a limited excerpt rather than sending an entire article.
    query = re.sub(r"\s+", " ", text)[:500]

    try:
        response = requests.get(
            FACT_CHECK_API_URL,
            params={
                "key": FACT_CHECK_API_KEY,
                "query": query,
                "languageCode": "en",
                "pageSize": 5,
            },
            timeout=12,
        )

        if response.status_code == 400:
            return jsonify({
                "available": False,
                "results": [],
                "message": (
                    "The fact-check search request was rejected. "
                    "Check that the Fact Check Tools API is enabled and "
                    "the API key is configured correctly."
                )
            }), 502

        if response.status_code in (401, 403):
            return jsonify({
                "available": False,
                "results": [],
                "message": (
                    "Google did not authorize the fact-check request. "
                    "Check the API key restrictions and API enablement."
                )
            }), 502

        response.raise_for_status()
        payload = response.json()

        results = []
        for claim in payload.get("claims", []):
            reviews = claim.get("claimReview", []) or []

            # A claim can have more than one published review.
            for review in reviews:
                url = review.get("url", "")
                if not isinstance(url, str) or not url.startswith(("https://", "http://")):
                    continue

                results.append({
                    "claim": claim.get("text", ""),
                    "claimant": claim.get("claimant", ""),
                    "publisher": review.get("publisher", {}).get("name", ""),
                    "title": review.get("title", ""),
                    "review_date": review.get("reviewDate", ""),
                    "rating": review.get("textualRating", ""),
                    "url": url,
                    "language": review.get("languageCode", ""),
                })

                if len(results) >= 10:
                    break

            if len(results) >= 10:
                break

        return jsonify({
            "available": True,
            "results": results,
            "message": (
                "Published fact-check reviews were found. Open the original "
                "reviews and compare their exact claims with the article."
                if results else
                "No matching published fact-check reviews were found. "
                "This does not mean the article is true or false."
            ),
            "disclaimer": (
                "Search results may concern similar but different claims. "
                "Check the wording, date, context, and original review before "
                "drawing a conclusion."
            )
        })

    except requests.Timeout:
        return jsonify({
            "available": False,
            "results": [],
            "message": "Fact-check search timed out. Please try again."
        }), 504
    except requests.RequestException:
        return jsonify({
            "available": False,
            "results": [],
            "message": (
                "Could not reach the fact-check service. Check your internet "
                "connection and Google Cloud API settings."
            )
        }), 502
    except ValueError:
        return jsonify({
            "available": False,
            "results": [],
            "message": "The fact-check service returned an unreadable response."
        }), 502


# ---------- DASHBOARD STATS ----------

@app.route("/api/stats")
def api_stats():
    with get_db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM analyses"
        ).fetchone()[0]

        fake_count = conn.execute(
            "SELECT COUNT(*) FROM analyses WHERE prediction = 'FAKE'"
        ).fetchone()[0]

        real_count = conn.execute(
            "SELECT COUNT(*) FROM analyses WHERE prediction = 'REAL'"
        ).fetchone()[0]

        average_confidence = conn.execute(
            "SELECT AVG(confidence) FROM analyses"
        ).fetchone()[0]

    return jsonify({
        "total_analyses": total,
        "fake_count": fake_count,
        "real_count": real_count,
        "average_confidence": (
            round(average_confidence, 2)
            if average_confidence is not None
            else None
        ),
    })


# ---------- RECENT HISTORY ----------

@app.route("/api/history")
def api_history():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, news_text, prediction, confidence,
                   confidence_level, suspicious_phrases,
                   word_count, created_at
            FROM analyses
            ORDER BY id DESC
            LIMIT 100
        """).fetchall()

    return jsonify([dict(row) for row in rows])


@app.route("/api/history/<int:analysis_id>", methods=["DELETE"])
def delete_analysis(analysis_id):
    with get_db() as conn:
        cursor = conn.execute(
            "DELETE FROM analyses WHERE id = ?",
            (analysis_id,),
        )

    if cursor.rowcount == 0:
        return jsonify({
            "error": "Analysis not found."
        }), 404

    return jsonify({"message": "Analysis deleted."})


init_db()

if __name__ == "__main__":
    # Local development only.
    app.run(debug=True)
