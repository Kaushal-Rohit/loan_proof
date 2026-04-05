# Loan Approval Predictor

An optimized machine-learning model with a Flask web application that predicts whether a loan applicant is eligible for approval.

---

## Features

| Area | Details |
|------|---------|
| **ML Models** | Random Forest, Gradient Boosting, XGBoost (ensemble) |
| **Tuning** | `RandomizedSearchCV` with Stratified K-Fold cross-validation |
| **Imbalance** | SMOTE oversampling (when `imbalanced-learn` is installed) |
| **Engineering** | Engineered ratio features (Income-to-Loan, Savings-to-Loan, etc.) |
| **Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Web App** | Flask with Bootstrap 5 – real-time prediction, dashboard, history |

---

## Project Structure

```
loan_proof/
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── config.py              # Configuration / paths
├── requirements.txt
├── templates/
│   ├── index.html         # Prediction form
│   ├── dashboard.html     # Model metrics
│   └── history.html       # Prediction history
├── static/
│   ├── css/style.css
│   └── js/main.js
├── models/                # Generated after training (gitignored)
└── data/                  # Place loan_approval_data.csv here (or use synthetic)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Add your data

Place `loan_approval_data.csv` inside the `data/` folder.  
If the file is absent, the app automatically generates a synthetic dataset for demonstration.

### 3. Train the model

```bash
python train_model.py
```

Trained model artifacts are saved to `models/`.

### 4. Run the web application

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

> The app also auto-trains on first launch if `models/best_model.pkl` is not found.

---

## Pages

| Route | Description |
|-------|-------------|
| `/` | Prediction form |
| `/dashboard` | Model metrics & feature importance chart |
| `/history` | Recent prediction history |

---

## Data Format

The CSV is expected to have these columns:

`Applicant_ID`, `Gender`, `Age`, `Education_Level`, `Marital_Status`,
`Employment_Status`, `Employer_Category`, `Applicant_Income`,
`Coapplicant_Income`, `Credit_Score`, `Loan_Amount`, `DTI_Ratio`,
`Savings`, `Existing_Loans`, `Dependents`, `Collateral_Value`,
`Property_Area`, `Loan_Purpose`, `Loan_Approved`

