# ============================================================
# train_model.py – Credit Risk Prediction Model Trainer
# ============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ───────────────────────────────────────────────────────────
print("=" * 60)
print("Loading credit risk dataset...")
print("=" * 60)

df = pd.read_csv("credit_risk_dataset.csv")
print(f"Dataset Shape: {df.shape}")
print(f"\nColumn Info:")
print(df.info())
print(f"\nFirst few rows:")
print(df.head())

# ───────────────────────────────────────────────────────────
# 2. HANDLE MISSING VALUES
# ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Handling missing values...")
print("=" * 60)

missing_counts = df.isnull().sum()
if missing_counts.sum() > 0:
    print("Missing values found:")
    print(missing_counts[missing_counts > 0])
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print("Missing values handled!")
else:
    print("No missing values found.")

# ───────────────────────────────────────────────────────────
# 3. IDENTIFY AND ENCODE CATEGORICAL FEATURES
# ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Encoding categorical features...")
print("=" * 60)

encoders = {}
categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

for col in categorical_features:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"✓ {col}: {len(le.classes_)} unique values")

# ───────────────────────────────────────────────────────────
# 4. PREPARE FEATURES AND TARGET
# ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Preparing features and target...")
print("=" * 60)

# Define features (exclude target)
FEATURES = [col for col in df.columns if col != 'loan_status']
X = df[FEATURES]
y = df['loan_status']

print(f"Features ({len(FEATURES)}): {FEATURES}")
print(f"Target: loan_status")
print(f"Class distribution:\n{y.value_counts()}")

# ───────────────────────────────────────────────────────────
# 5. TRAIN/TEST SPLIT
# ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Splitting dataset...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ───────────────────────────────────────────────────────────
# 6. TRAIN MODEL
# ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Training Random Forest Classifier...")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

model.fit(X_train, y_train)
print("✓ Model training completed!")

# ───────────────────────────────────────────────────────────
# 7. EVALUATE MODEL
# ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Model Evaluation")
print("=" * 60)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")
print(f"\nDetailed Test Metrics:")
print(f"Precision: {precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred, zero_division=0):.4f}")

print("\n" + "-" * 60)
print("Classification Report:")
print("-" * 60)
print(classification_report(y_test, y_test_pred))

print("\n" + "-" * 60)
print("Confusion Matrix:")
print("-" * 60)
print(confusion_matrix(y_test, y_test_pred))

# Feature importance
print("\n" + "=" * 60)
print("Top 10 Feature Importances")
print("=" * 60)
feature_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# ───────────────────────────────────────────────────────────
# 8. SAVE MODEL ARTIFACT
# ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Saving Model Artifact")
print("=" * 60)

artifact = {
    "model": model,
    "encoders": encoders,
    "features": FEATURES,
    "categorical_features": categorical_features,
    "feature_importance": feature_importance.to_dict('records'),
    "metrics": {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_test_pred, zero_division=0)
    }
}

with open("model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("✓ model.pkl saved successfully!")
print("\n" + "=" * 60)
print("Ready to run: python app.py")
print("=" * 60)