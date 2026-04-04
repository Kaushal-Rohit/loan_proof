import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "predictions.db")

DATA_FILE = os.path.join(DATA_DIR, "loan_approval_data.csv")

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER_SEARCH = 30

# Feature definitions
CATEGORICAL_OHE = ["Gender", "Employer_Category", "Employment_Status"]
CATEGORICAL_LE = ["Education_Level", "Marital_Status", "Property_Area", "Loan_Purpose"]
NUMERICAL_FEATURES = [
    "Age",
    "Applicant_Income",
    "Coapplicant_Income",
    "Credit_Score",
    "Loan_Amount",
    "DTI_Ratio",
    "Savings",
    "Existing_Loans",
    "Dependents",
    "Collateral_Value",
]
TARGET = "Loan_Approved"
DROP_COLS = ["Applicant_ID"]
