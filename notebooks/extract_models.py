"""
Extract and save trained models
Run once to create .pkl files
"""

import pandas as pd
import joblib
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ==================== CREATE MODELS DIRECTORY ====================

os.makedirs("models", exist_ok=True)

# ==================== LOAD DATA ====================

df = pd.read_csv("loan_approval_data.csv")

# Create working copy
df_clean = df.copy()

# ==================== HANDLE MISSING VALUES ====================

# Categorical columns
categorical_fill = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Gender",
    "Employer_Category"
]

for col in categorical_fill:
    df_clean[col] = df_clean[col].fillna(
        df_clean[col].mode()[0]
    )

# Numerical columns
numerical_fill = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Dependents",
    "Credit_Score",
    "Existing_Loans",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term"
]

for col in numerical_fill:
    df_clean[col] = df_clean[col].fillna(
        df_clean[col].mean()
    )

# Remove rows where target is missing
df_clean = df_clean.dropna(subset=["Loan_Approved"])

# ==================== REMOVE ID COLUMN ====================

if "Applicant_ID" in df_clean.columns:
    df_clean = df_clean.drop("Applicant_ID", axis=1)

# ==================== FEATURES & TARGET ====================

X = df_clean.drop("Loan_Approved", axis=1)

y = df_clean["Loan_Approved"].map({
    "Yes": 1,
    "No": 0
})

# ==================== ENCODE CATEGORICAL FEATURES ====================

categorical_cols = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

label_encoders = {}

X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()

    X_encoded[col] = le.fit_transform(
        X_encoded[col].astype(str)
    )

    label_encoders[col] = le

# ==================== SCALE FEATURES ====================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_encoded)

# ==================== TRAIN TEST SPLIT ====================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ==================== TRAIN MODELS ====================

# Logistic Regression
log_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

log_model.fit(X_train, y_train)

joblib.dump(
    log_model,
    "models/logistic_regression_model.pkl"
)

print("✓ Logistic Regression model saved")

# Naive Bayes
nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

joblib.dump(
    nb_model,
    "models/naive_bayes_model.pkl"
)

print("✓ Naive Bayes model saved")

# KNN
knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(X_train, y_train)

joblib.dump(
    knn_model,
    "models/knn_model.pkl"
)

print("✓ KNN model saved")

# ==================== SAVE PREPROCESSING OBJECTS ====================

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("✓ Scaler saved")

joblib.dump(
    label_encoders,
    "models/label_encoders.pkl"
)

print("✓ Label encoders saved")

feature_names = X_encoded.columns.tolist()

joblib.dump(
    feature_names,
    "models/feature_names.pkl"
)

print("✓ Feature names saved")

# ==================== QUICK EVALUATION ====================

y_pred = nb_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 50)
print("All models saved successfully!")
print("=" * 50)
print(f"\nNaive Bayes Accuracy: {accuracy:.2%}")
print("Ready for Streamlit deployment 🚀")