# ============================================================
# app.py  –  Loan Eligibility Prediction  |  Flask Backend
# Routes:
#   GET  /          → Render the prediction form
#   POST /predict   → Accept form data, return prediction
# ============================================================

from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# ─── Load the trained model artifact ─────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

model    = artifact["model"]
encoders = artifact["encoders"]
FEATURES = artifact["features"]

# ─── Helper: safely encode a value with a LabelEncoder ──────
def safe_encode(encoder, value):
    """Return encoded integer; fall back to 0 for unseen labels."""
    classes = list(encoder.classes_)
    return classes.index(value) if value in classes else 0


# ─── Route: Home ─────────────────────────────────────────────
@app.route("/")
def index():
    """Render the loan application form."""
    return render_template("index.html", prediction=None)


# ─── Route: Predict ──────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Read form values → encode → run model → show result.
    Expected form fields (all strings from the HTML form):
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income, loan_amount,
        loan_amount_term, credit_history
    """
    try:
        # ── Collect raw form values ──────────────────────────
        gender           = request.form.get("gender", "Male")
        married          = request.form.get("married", "No")
        dependents_raw   = request.form.get("dependents", "0")
        education        = request.form.get("education", "Graduate")
        self_employed    = request.form.get("self_employed", "No")
        applicant_income = float(request.form.get("applicant_income", 0))
        coapplicant_income = float(request.form.get("coapplicant_income", 0))
        loan_amount      = float(request.form.get("loan_amount", 0))
        loan_amount_term = float(request.form.get("loan_amount_term", 360))
        credit_history   = float(request.form.get("credit_history", 1))

        # ── Encode categorical values ────────────────────────
        gender_enc      = safe_encode(encoders["Gender"],      gender)
        married_enc     = safe_encode(encoders["Married"],     married)
        education_enc   = safe_encode(encoders["Education"],   education)
        self_emp_enc    = safe_encode(encoders["Self_Employed"], self_employed)

        # Dependents: convert "3+" → 3
        dependents = int(dependents_raw.replace("3+", "3"))

        # ── Build feature vector (same order as FEATURES) ────
        features = np.array([[
            gender_enc,
            married_enc,
            dependents,
            education_enc,
            self_emp_enc,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history
        ]])

        # ── Run prediction ───────────────────────────────────
        prediction_code = model.predict(features)[0]
        probability     = model.predict_proba(features)[0]

        # encoders["Loan_Status"] classes: [N=0, Y=1]
        if prediction_code == 1:
            result   = "Loan Approved"
            approved = True
            conf     = round(probability[1] * 100, 1)
        else:
            result   = "Loan Not Approved"
            approved = False
            conf     = round(probability[0] * 100, 1)

        return render_template(
            "index.html",
            prediction=result,
            approved=approved,
            confidence=conf,
            # Pass back form data so the form stays filled
            form_data=request.form
        )

    except Exception as e:
        # Surface any error in the template for easier debugging
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}",
            approved=False,
            confidence=None,
            form_data=request.form
        )


# ─── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)