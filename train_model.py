
import nt

import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# 1. Load datasets
# -----------------------------
print("Loading datasets...")

fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

fake_df["label"] = "FAKE"
real_df["label"] = "REAL"

df = pd.concat([fake_df, real_df], ignore_index=True)

# -----------------------------
# 2. Clean data
# -----------------------------
print("Cleaning dataset...")

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

df["content"] = (
    df["title"].astype(str) + " " +
    df["text"].astype(str)
).str.strip()

df = df[df["content"] != ""]
df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)

# Remove source-style prefixes sometimes found in the text
df["content"] = df["content"].apply(
    lambda text: re.sub(
        r"^\s*\[[^\]]+\]\s*",
        "",
        text
    )
)

X = df["content"]
y = df["label"]

print(f"Total articles: {len(df)}")
print("\nDataset distribution:")
print(y.value_counts())

# -----------------------------
# 3. Split into train, validation and test
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Testing samples: {len(X_test)}")

# -----------------------------
# 4. Build model with C = 0.1
# -----------------------------
print("\nTraining Logistic Regression model...")
print("Selected C value: 0.1")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=100000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=42
        )
    )
])

# Train on training data
model.fit(X_train, y_train)

# -----------------------------
# 5. Validation evaluation
# -----------------------------
val_predictions = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)

print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%")

# -----------------------------
# 6. Final test evaluation
# -----------------------------
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=["FAKE", "REAL"],
        zero_division=0
    )
)

print("Confusion Matrix (rows = actual, columns = predicted):")
print(
    confusion_matrix(
        y_test,
        test_predictions,
        labels=["FAKE", "REAL"]
    )
)

# -----------------------------
# 7. Export incorrect predictions
# -----------------------------
print("\nExporting incorrect predictions...")

error_df = pd.DataFrame({
    "actual": y_test,
    "predicted": test_predictions,
    "content": X_test
})

errors = error_df[
    error_df["actual"] != error_df["predicted"]
].copy()

errors.to_csv("prediction_errors.csv", index=False)

print(f"Incorrect predictions: {len(errors)}")
print("Saved to prediction_errors.csv")

print("\nSample incorrect predictions:")
for _, row in errors.head(10).iterrows():
    print("-" * 60)
    print("Actual:", row["actual"])
    print("Predicted:", row["predicted"])
    print("Article:", str(row["content"])[:500])

# -----------------------------
# 8. Save trained model
# -----------------------------
joblib.dump(model, "fake_news_model.joblib")

print("\nModel saved as fake_news_model.joblib")
print("Training and testing completed!")
