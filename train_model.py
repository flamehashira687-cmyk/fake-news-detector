
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent

fake_path = BASE_DIR / "Fake.csv"
true_path = BASE_DIR / "True.csv"

if not fake_path.exists() or not true_path.exists():
    raise FileNotFoundError(
        "Fake.csv and True.csv must be in the project folder."
    )

print("Loading datasets...")
fake = pd.read_csv(fake_path)
real = pd.read_csv(true_path)

fake["label"] = "FAKE"
real["label"] = "REAL"

df = pd.concat([fake, real], ignore_index=True)

# Combine headline and article text.
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")
df["content"] = (
    df["title"].astype(str) + " " + df["text"].astype(str)
).str.strip()

df = df[df["content"].str.len() > 0].copy()
df = df.drop_duplicates(subset=["content"])

# Remove exact Reuters source markers that can act as shortcuts.
# The model still learns from the remaining article text.
df["content"] = df["content"].str.replace(
    r"(?i)^\s*[^.]{0,80}\s*-\s*",
    "",
    regex=True,
)

X = df["content"]
y = df["label"]

print("Dataset size:", len(df))
print("Label counts:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=100000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
        C=0.01,
        max_iter=1000,
        random_state=42,
        ),
    ),
])

print("\nTraining model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nTest accuracy:", round(
    accuracy_score(y_test, predictions), 4
))
print("\nClassification report:")
print(classification_report(y_test, predictions))
print("Confusion matrix [FAKE, REAL]:")
print(confusion_matrix(
    y_test, predictions, labels=["FAKE", "REAL"]
))

model_path = BASE_DIR / "fake_news_model.joblib"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)
print("Training complete!")
