"""
Optimized loan approval prediction model training script.

Steps:
  1. Load data (or generate synthetic data for demo purposes)
  2. Feature engineering
  3. Build preprocessing + model pipeline
  4. Hyperparameter tuning with RandomizedSearchCV
  5. Ensemble with VotingClassifier
  6. Evaluate with cross-validation metrics
  7. Save model artifacts
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

warnings.filterwarnings("ignore")

import config


# ---------------------------------------------------------------------------
# Synthetic data generator (used when the real CSV is not present)
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 66, n_samples)
    applicant_income = rng.integers(15000, 200001, n_samples)
    coapplicant_income = rng.integers(0, 100001, n_samples)
    credit_score = rng.integers(300, 851, n_samples)
    loan_amount = rng.integers(50000, 500001, n_samples)
    dti_ratio = rng.uniform(0.05, 0.80, n_samples).round(3)
    savings = rng.integers(0, 500001, n_samples)
    existing_loans = rng.integers(0, 6, n_samples)
    dependents = rng.integers(0, 6, n_samples)
    collateral_value = rng.integers(0, 1000001, n_samples)

    gender = rng.choice(["Male", "Female"], n_samples)
    education = rng.choice(["Graduate", "Not Graduate"], n_samples, p=[0.6, 0.4])
    marital_status = rng.choice(["Married", "Single", "Divorced"], n_samples, p=[0.55, 0.35, 0.10])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], n_samples)
    loan_purpose = rng.choice(["Home", "Education", "Business", "Personal", "Auto"], n_samples)
    employer_category = rng.choice(
        ["Government", "Private", "Self-Employed", "Business Owner"], n_samples
    )
    employment_status = rng.choice(
        ["Employed", "Self-Employed", "Unemployed"], n_samples, p=[0.65, 0.25, 0.10]
    )

    # Realistic approval logic (score centred at SCORE_THRESHOLD, spread SCORE_SCALE)
    SCORE_THRESHOLD = 45
    SCORE_SCALE = 10
    score = (
        (credit_score - 300) / 550 * 30
        + (applicant_income / 200000) * 20
        + (savings / 500000) * 15
        + ((1 - dti_ratio) * 15)
        + (collateral_value / 1000000) * 10
        + (education == "Graduate").astype(int) * 5
        + (employment_status == "Employed").astype(int) * 5
        - existing_loans * 2
        - dependents * 1
    )
    prob = 1 / (1 + np.exp(-(score - SCORE_THRESHOLD) / SCORE_SCALE))
    loan_approved = rng.random(n_samples) < prob

    df = pd.DataFrame(
        {
            "Applicant_ID": [f"APP{i:05d}" for i in range(n_samples)],
            "Gender": gender,
            "Age": age,
            "Education_Level": education,
            "Marital_Status": marital_status,
            "Employment_Status": employment_status,
            "Employer_Category": employer_category,
            "Applicant_Income": applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Credit_Score": credit_score,
            "Loan_Amount": loan_amount,
            "DTI_Ratio": dti_ratio,
            "Savings": savings,
            "Existing_Loans": existing_loans,
            "Dependents": dependents,
            "Collateral_Value": collateral_value,
            "Property_Area": property_area,
            "Loan_Purpose": loan_purpose,
            "Loan_Approved": np.where(loan_approved, "Yes", "No"),
        }
    )
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Small epsilon to avoid division by zero in ratio features
    EPSILON = 1
    df = df.copy()
    df["Total_Income"] = df["Applicant_Income"] + df["Coapplicant_Income"]
    df["Income_to_Loan_Ratio"] = df["Total_Income"] / (df["Loan_Amount"] + EPSILON)
    df["Loan_to_Collateral_Ratio"] = df["Loan_Amount"] / (df["Collateral_Value"] + EPSILON)
    df["Savings_to_Loan_Ratio"] = df["Savings"] / (df["Loan_Amount"] + EPSILON)
    df["Credit_Score_Squared"] = df["Credit_Score"] ** 2
    df["DTI_Squared"] = df["DTI_Ratio"] ** 2
    return df


# ---------------------------------------------------------------------------
# Build preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessor(X: pd.DataFrame):
    numerical_features = [
        c for c in X.columns
        if X[c].dtype in [np.float64, np.int64, np.int32, np.float32]
    ]
    ohe_features = [c for c in config.CATEGORICAL_OHE if c in X.columns]
    ordinal_features = [c for c in config.CATEGORICAL_LE if c in X.columns]

    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    ohe_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
            ),
        ]
    )

    ordinal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("ohe", ohe_pipeline, ohe_features),
            ("ord", ordinal_pipeline, ordinal_features),
        ],
        remainder="drop",
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train():
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # ---- Load data --------------------------------------------------------
    if os.path.exists(config.DATA_FILE):
        print(f"[INFO] Loading data from {config.DATA_FILE}")
        df = pd.read_csv(config.DATA_FILE)
    else:
        print("[INFO] Data file not found – generating synthetic dataset for demo.")
        df = generate_synthetic_data()
        df.to_csv(config.DATA_FILE, index=False)
        print(f"[INFO] Synthetic data saved to {config.DATA_FILE}")

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Class distribution:\n{df[config.TARGET].value_counts()}")

    # ---- Drop ID columns ---------------------------------------------------
    for col in config.DROP_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # ---- Feature engineering -----------------------------------------------
    df = engineer_features(df)

    # ---- Encode target -----------------------------------------------------
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[config.TARGET])
    X = df.drop(columns=[config.TARGET])

    print(f"[INFO] Features used: {list(X.columns)}")

    # ---- Train / test split -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    # ---- Preprocessor -------------------------------------------------------
    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # ---- Handle class imbalance with SMOTE ----------------------------------
    if HAS_SMOTE:
        smote = SMOTE(random_state=config.RANDOM_STATE)
        X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)
        print(f"[INFO] After SMOTE – training samples: {X_train_proc.shape[0]}")

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    # ---- Hyperparameter search for Random Forest ----------------------------
    print("[INFO] Tuning Random Forest …")
    rf_param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", None],
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=config.RANDOM_STATE),
        rf_param_dist,
        n_iter=config.N_ITER_SEARCH,
        scoring="roc_auc",
        cv=cv,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    rf_search.fit(X_train_proc, y_train)
    best_rf = rf_search.best_estimator_
    print(f"  Best RF params: {rf_search.best_params_}")
    print(f"  Best RF CV AUC: {rf_search.best_score_:.4f}")

    # ---- Hyperparameter search for Gradient Boosting ------------------------
    print("[INFO] Tuning Gradient Boosting …")
    gb_param_dist = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
        "min_samples_split": [2, 5],
    }
    gb_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=config.RANDOM_STATE),
        gb_param_dist,
        n_iter=config.N_ITER_SEARCH,
        scoring="roc_auc",
        cv=cv,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    gb_search.fit(X_train_proc, y_train)
    best_gb = gb_search.best_estimator_
    print(f"  Best GB params: {gb_search.best_params_}")
    print(f"  Best GB CV AUC: {gb_search.best_score_:.4f}")

    # ---- XGBoost (optional) -------------------------------------------------
    estimators_for_ensemble = [("rf", best_rf), ("gb", best_gb)]

    if HAS_XGB:
        print("[INFO] Tuning XGBoost …")
        xgb_param_dist = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        }
        xgb_search = RandomizedSearchCV(
            XGBClassifier(
                random_state=config.RANDOM_STATE,
                eval_metric="logloss",
                verbosity=0,
            ),
            xgb_param_dist,
            n_iter=config.N_ITER_SEARCH,
            scoring="roc_auc",
            cv=cv,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        xgb_search.fit(X_train_proc, y_train)
        best_xgb = xgb_search.best_estimator_
        print(f"  Best XGB params: {xgb_search.best_params_}")
        print(f"  Best XGB CV AUC: {xgb_search.best_score_:.4f}")
        estimators_for_ensemble.append(("xgb", best_xgb))

    # ---- Logistic Regression (baseline) ------------------------------------
    lr = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
    lr.fit(X_train_proc, y_train)
    lr_auc = cross_val_score(lr, X_train_proc, y_train, cv=cv, scoring="roc_auc").mean()
    print(f"[INFO] Logistic Regression CV AUC: {lr_auc:.4f}")

    # ---- Ensemble -----------------------------------------------------------
    print("[INFO] Building Voting Ensemble …")
    ensemble = VotingClassifier(estimators=estimators_for_ensemble, voting="soft", n_jobs=-1)
    ensemble.fit(X_train_proc, y_train)

    ensemble_cv = cross_val_score(
        ensemble, X_train_proc, y_train, cv=cv, scoring="roc_auc"
    ).mean()
    print(f"[INFO] Ensemble CV AUC: {ensemble_cv:.4f}")

    # ---- Evaluate on test set -----------------------------------------------
    y_pred = ensemble.predict(X_test_proc)
    y_proba = ensemble.predict_proba(X_test_proc)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "cv_auc": float(ensemble_cv),
    }
    print("\n[RESULTS] Test-set metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n[RESULTS] Classification Report:")
    print(classification_report(y_test, y_pred))

    # ---- Feature importances ------------------------------------------------
    feature_names = []
    try:
        num_names = list(preprocessor.transformers_[0][2])
        ohe_names = list(
            preprocessor.named_transformers_["ohe"]
            .named_steps["encoder"]
            .get_feature_names_out(config.CATEGORICAL_OHE)
        )
        ord_names = [c for c in config.CATEGORICAL_LE if c in X.columns]
        feature_names = num_names + ohe_names + ord_names
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train_proc.shape[1])]

    importances = None
    if hasattr(best_rf, "feature_importances_"):
        importances = best_rf.feature_importances_

    if importances is not None and len(importances) == len(feature_names):
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df = fi_df.sort_values("importance", ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="steelblue")
        plt.xlabel("Importance")
        plt.title("Top Feature Importances (Random Forest)")
        plt.tight_layout()
        plt.savefig(os.path.join(config.MODELS_DIR, "feature_importance.png"), dpi=100)
        plt.close()

        feature_importance_dict = dict(zip(fi_df["feature"], fi_df["importance"].round(4)))
    else:
        feature_importance_dict = {}

    # ---- Save artifacts ----------------------------------------------------
    joblib.dump(ensemble, os.path.join(config.MODELS_DIR, "best_model.pkl"))
    joblib.dump(preprocessor, os.path.join(config.MODELS_DIR, "preprocessor.pkl"))
    joblib.dump(le_target, os.path.join(config.MODELS_DIR, "label_encoder.pkl"))
    joblib.dump(list(X.columns), os.path.join(config.MODELS_DIR, "feature_names.pkl"))

    with open(os.path.join(config.MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(config.MODELS_DIR, "feature_importance.json"), "w") as f:
        json.dump(feature_importance_dict, f, indent=2)

    print(f"\n[INFO] Model artifacts saved to {config.MODELS_DIR}")
    print("[INFO] Training complete!")


if __name__ == "__main__":
    train()
