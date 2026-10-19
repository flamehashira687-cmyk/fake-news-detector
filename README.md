# TruthLens — AI Fake News Detection

## Overview
TruthLens is a web-based application that uses Machine Learning to analyze news articles and identify potentially fake or real news. It also provides prediction confidence, explainable AI insights, analysis history, dashboard statistics, and external fact-check source verification.

## Features
- **Fake News Detection:** TF-IDF and Logistic Regression.
- **Confidence Score:** Displays prediction confidence.
- **Explainable AI (XAI):** Highlights influential words in predictions.
- **Dashboard:** Displays analysis statistics and history.
- **History Management:** Stores and deletes previous analyses.
- **Fact-Check Verification:** Searches published fact-check reviews using the Google Fact Check Tools API.
- **Web Interface:** Interactive Flask-based application.

## Technology Stack
- Python
- Flask
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- SQLite
- HTML, CSS, JavaScript
- Google Fact Check Tools API

## Installation

1. Clone or download this repository.
2. Open the project directory.
3. Create and activate a virtual environment.
4. Install dependencies:

```bash
pip install flask scikit-learn pandas joblib requests python-dotenv
```

5. Configure your Google Fact Check API key in a `.env` file:

```env
GOOGLE_FACT_CHECK_API_KEY=your_api_key_here
```

6. Start the application:

```bash
python app.py
```

7. Open `http://127.0.0.1:5000` in your browser.

## Machine Learning Workflow
1. Load and preprocess the news dataset.
2. Convert text into numerical features using TF-IDF.
3. Train a Logistic Regression classifier.
4. Evaluate model performance.
5. Analyze news through the Flask application.
6. Display predictions, confidence scores, and explanations.

## Important Note
TruthLens provides predictions, not guaranteed truth judgments. Model confidence is not proof of accuracy, and an absence of matching fact-check results does not mean a claim is true or false. Always verify important claims using reliable sources.

## Security
- Never commit API keys or secrets.
- Keep `.env` in `.gitignore`.
- Do not publish private credentials.

## Project Status
Core analysis, dashboard, history management, and fact-check verification features have been implemented. Testing and documentation are ongoing.
