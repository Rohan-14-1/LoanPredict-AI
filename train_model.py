# ============================================================
# train_model.py – Loan Eligibility Prediction
# ============================================================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv("loan_data.csv")
print("Shape:", df.shape)

# 2. Drop unnecessary column
df.drop("Loan_ID", axis=1, inplace=True)

# 3. Handle Missing Values
categorical_cols = ["Gender", "Married", "Dependents",
                    "Self_Employed", "Credit_History", "Property_Area"]

numerical_cols = ["LoanAmount", "Loan_Amount_Term",
                  "ApplicantIncome", "CoapplicantIncome"]

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

print("Missing values handled")

# 4. Encode categorical
encoders = {}

label_cols = ["Gender", "Married", "Education",
              "Self_Employed", "Property_Area", "Loan_Status"]

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

df["Dependents"] = df["Dependents"].replace("3+", "3").astype(int)

print("Encoding done")

# 5. Features & Target
FEATURES = ["Gender", "Married", "Dependents", "Education",
            "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
            "LoanAmount", "Loan_Amount_Term", "Credit_History"]

X = df[FEATURES]
y = df["Loan_Status"]

# 6. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# 9. Save model
artifact = {
    "model": model,
    "encoders": encoders,
    "features": FEATURES
}

with open("model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("model.pkl saved successfully")
print("Run: python app.py")