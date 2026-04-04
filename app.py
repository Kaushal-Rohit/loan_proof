"""
Flask web application for loan approval prediction.
"""

import os
import json
import sqlite3
import traceback
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for

import config
from train_model import engineer_features, train

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model artifacts (train on first run if not found)
# ---------------------------------------------------------------------------

def load_artifacts():
    model_path = os.path.join(config.MODELS_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        print("[APP] No trained model found – training now …")
        train()

    model = joblib.load(os.path.join(config.MODELS_DIR, "best_model.pkl"))
    preprocessor = joblib.load(os.path.join(config.MODELS_DIR, "preprocessor.pkl"))
    label_encoder = joblib.load(os.path.join(config.MODELS_DIR, "label_encoder.pkl"))
    feature_names = joblib.load(os.path.join(config.MODELS_DIR, "feature_names.pkl"))

    metrics_path = os.path.join(config.MODELS_DIR, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    fi_path = os.path.join(config.MODELS_DIR, "feature_importance.json")
    feature_importance = {}
    if os.path.exists(fi_path):
        with open(fi_path) as f:
            feature_importance = json.load(f)

    return model, preprocessor, label_encoder, feature_names, metrics, feature_importance


MODEL, PREPROCESSOR, LABEL_ENCODER, FEATURE_NAMES, METRICS, FEATURE_IMPORTANCE = (
    load_artifacts()
)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db():
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            name        TEXT,
            prediction  TEXT    NOT NULL,
            probability REAL    NOT NULL,
            features    TEXT    NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction(name: str, prediction: str, probability: float, features: dict):
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(
        "INSERT INTO predictions (timestamp, name, prediction, probability, features) VALUES (?,?,?,?,?)",
        (
            datetime.utcnow().isoformat(timespec="seconds"),
            name or "Anonymous",
            prediction,
            round(probability, 4),
            json.dumps(features),
        ),
    )
    conn.commit()
    conn.close()


def get_recent_predictions(limit: int = 20):
    conn = sqlite3.connect(config.DB_PATH)
    cur = conn.execute(
        "SELECT id, timestamp, name, prediction, probability FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    conn.close()
    return rows


init_db()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

VALID_GENDERS = {"Male", "Female"}
VALID_EDUCATION = {"Graduate", "Not Graduate"}
VALID_MARITAL = {"Married", "Single", "Divorced"}
VALID_PROPERTY = {"Urban", "Semiurban", "Rural"}
VALID_PURPOSE = {"Home", "Education", "Business", "Personal", "Auto"}
VALID_EMPLOYER = {"Government", "Private", "Self-Employed", "Business Owner"}
VALID_EMPLOYMENT = {"Employed", "Self-Employed", "Unemployed"}


def validate_and_parse(form: dict) -> tuple[dict, list]:
    errors = []
    data = {}

    def _float(key, label, lo, hi):
        try:
            v = float(form.get(key, ""))
            if not (lo <= v <= hi):
                errors.append(f"{label} must be between {lo} and {hi}.")
            else:
                data[key] = v
        except (TypeError, ValueError):
            errors.append(f"{label} must be a valid number.")

    def _int(key, label, lo, hi):
        try:
            v = int(form.get(key, ""))
            if not (lo <= v <= hi):
                errors.append(f"{label} must be between {lo} and {hi}.")
            else:
                data[key] = v
        except (TypeError, ValueError):
            errors.append(f"{label} must be a valid integer.")

    def _choice(key, label, choices):
        v = form.get(key, "")
        if v not in choices:
            errors.append(f"{label} must be one of: {', '.join(sorted(choices))}.")
        else:
            data[key] = v

    _int("Age", "Age", 18, 100)
    _float("Applicant_Income", "Applicant Income", 0, 10_000_000)
    _float("Coapplicant_Income", "Co-applicant Income", 0, 10_000_000)
    _float("Credit_Score", "Credit Score", 300, 850)
    _float("Loan_Amount", "Loan Amount", 1000, 100_000_000)
    _float("DTI_Ratio", "DTI Ratio", 0, 1)
    _float("Savings", "Savings", 0, 100_000_000)
    _int("Existing_Loans", "Existing Loans", 0, 20)
    _int("Dependents", "Dependents", 0, 20)
    _float("Collateral_Value", "Collateral Value", 0, 100_000_000)
    _choice("Gender", "Gender", VALID_GENDERS)
    _choice("Education_Level", "Education Level", VALID_EDUCATION)
    _choice("Marital_Status", "Marital Status", VALID_MARITAL)
    _choice("Property_Area", "Property Area", VALID_PROPERTY)
    _choice("Loan_Purpose", "Loan Purpose", VALID_PURPOSE)
    _choice("Employer_Category", "Employer Category", VALID_EMPLOYER)
    _choice("Employment_Status", "Employment Status", VALID_EMPLOYMENT)

    return data, errors


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        genders=sorted(VALID_GENDERS),
        education_levels=sorted(VALID_EDUCATION),
        marital_statuses=sorted(VALID_MARITAL),
        property_areas=sorted(VALID_PROPERTY),
        loan_purposes=sorted(VALID_PURPOSE),
        employer_categories=sorted(VALID_EMPLOYER),
        employment_statuses=sorted(VALID_EMPLOYMENT),
    )


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form.to_dict()
    applicant_name = form.pop("applicant_name", "").strip()

    data, errors = validate_and_parse(form)
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    try:
        df = pd.DataFrame([data])
        df = engineer_features(df)

        # Reorder to match training feature order
        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0

        df = df[FEATURE_NAMES]

        X_proc = PREPROCESSOR.transform(df)
        pred_idx = MODEL.predict(X_proc)[0]
        proba = MODEL.predict_proba(X_proc)[0]

        prediction = LABEL_ENCODER.inverse_transform([pred_idx])[0]
        probability = float(proba[pred_idx])

        save_prediction(applicant_name, prediction, probability, data)

        return jsonify(
            {
                "success": True,
                "prediction": prediction,
                "probability": round(probability * 100, 2),
                "approved": prediction == "Yes",
            }
        )
    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "errors": ["Internal server error."]}), 500


@app.route("/dashboard")
def dashboard():
    fi_items = sorted(
        FEATURE_IMPORTANCE.items(), key=lambda x: x[1], reverse=True
    )[:15]
    fi_labels = [i[0] for i in fi_items]
    fi_values = [round(i[1] * 100, 2) for i in fi_items]

    return render_template(
        "dashboard.html",
        metrics=METRICS,
        fi_labels=json.dumps(fi_labels),
        fi_values=json.dumps(fi_values),
    )


@app.route("/history")
def history():
    rows = get_recent_predictions()
    return render_template("history.html", predictions=rows)


@app.route("/history/clear", methods=["POST"])
def clear_history():
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return redirect(url_for("history"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
