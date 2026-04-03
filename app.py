# ============================================================
# app.py – Loan Eligibility Prediction | Flask Backend
# ============================================================

from flask import Flask, render_template, request, jsonify
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
    classes = list(encoder.classes_)
    return classes.index(value) if value in classes else 0


# ─── Route: Home ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ─── Route: Analysis Page ───────────────────────────────────
@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


# ─── Route: Dashboard Page ──────────────────────────────────
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ─── Route: Simulator Page ──────────────────────────────────
@app.route("/simulator")
def simulator():
    return render_template("simulator.html")


# ─── Route: Navbar Fragment ─────────────────────────────────
@app.route("/navbar")
def navbar():
    return render_template("navbar.html")


# ─── Route: Process Application (JSON API) ──────────────────
@app.route("/process_application", methods=["POST"])
def process_application():
    """
    Accepts JSON from the new frontend analysis form.
    Maps incoming fields to the trained model's expected features
    and returns a JSON prediction result.
    """
    try:
        data = request.get_json()

        # ── Extract values from the new frontend payload ─────
        monthly_income   = float(data.get("income", 0))
        loan_amount_raw  = float(data.get("loan_amount", 0))
        loan_term        = float(data.get("loan_term", 360))
        cibil_score      = float(data.get("cibil_score", 750))
        missed_payments  = int(data.get("missed_payments", 0))

        # ── Map to model features ────────────────────────────
        # Categorical defaults (not collected by new frontend)
        gender_enc    = safe_encode(encoders["Gender"], "Male")
        married_enc   = safe_encode(encoders["Married"], "Yes")
        education_enc = safe_encode(encoders["Education"], "Graduate")
        self_emp_enc  = safe_encode(encoders["Self_Employed"], "No")
        dependents    = 0

        # Numeric mappings
        applicant_income   = monthly_income          # monthly income
        coapplicant_income = 0                       # not collected
        loan_amount        = loan_amount_raw / 1000  # model trained on ₹K
        loan_amount_term   = loan_term               # in months

        # Credit history: good if no missed payments, else bad
        credit_history = 1.0 if missed_payments == 0 else 0.0

        total_income = applicant_income + coapplicant_income

        # ── Rule 1: No income → Reject ───────────────────────
        if total_income <= 0:
            return jsonify({
                "status": "rejected",
                "odds": 0,
                "main_issue": "No income reported",
                "coach_advice": "The applicant has reported zero income. Loan cannot be approved without verified income sources."
            })

        # ── Rule 2: Loan too high compared to income → Reject
        if loan_amount_raw > total_income * 20:
            return jsonify({
                "status": "rejected",
                "odds": 5,
                "main_issue": "Loan amount exceeds income capacity",
                "coach_advice": "The requested loan amount is disproportionately high relative to the applicant's income. Consider reducing the loan amount or increasing income documentation."
            })

        # ── Feature array ─────────────────────────────────────
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

        # ── Prediction ────────────────────────────────────────
        prediction_code = model.predict(features)[0]
        probability     = model.predict_proba(features)[0]

        bank_offers = []

        if prediction_code == 1:
            status = "approved"
            odds   = round(probability[1] * 100, 1)

            # Bank suggestions based on confidence
            if odds >= 85:
                coach_advice = "Excellent financial profile. Strong income stability, clean credit history, and manageable loan burden. This borrower is recommended for premium interest rates."
                bank_offers = [
                    {"name": "State Bank of India", "interest": "8.25%", "type": "Public Sector", "icon": "building-columns"},
                    {"name": "HDFC Bank", "interest": "8.50%", "type": "Private Sector", "icon": "landmark"},
                    {"name": "ICICI Bank", "interest": "8.65%", "type": "Private Sector", "icon": "building"},
                    {"name": "Bank of Baroda", "interest": "8.40%", "type": "Public Sector", "icon": "university"}
                ]
            elif odds >= 70:
                coach_advice = "Good financial profile with acceptable risk levels. Credit meets minimum thresholds. Monitor debt obligations and consider additional collateral for better rates."
                bank_offers = [
                    {"name": "Axis Bank", "interest": "9.25%", "type": "Private Sector", "icon": "building"},
                    {"name": "Kotak Mahindra Bank", "interest": "9.50%", "type": "Private Sector", "icon": "landmark"},
                    {"name": "Punjab National Bank", "interest": "8.90%", "type": "Public Sector", "icon": "building-columns"}
                ]
            else:
                coach_advice = "Borderline approval. While the model predicts approval, the confidence is moderate. Consider strengthening the application with co-applicant income or additional documentation."
                bank_offers = [
                    {"name": "NBFC / Private Lenders", "interest": "11.5%", "type": "NBFC", "icon": "hand-holding-dollar"},
                    {"name": "Bajaj Finserv", "interest": "12.0%", "type": "NBFC", "icon": "coins"}
                ]
        else:
            status = "rejected"
            odds   = round(probability[0] * 100, 1)

            if credit_history == 0:
                main_issue = "Poor credit history"
                coach_advice = "Credit history does not meet lending guidelines. This is the primary factor in the rejection. Recommend building credit history and clearing outstanding obligations before reapplying."
            elif loan_amount_raw > total_income * 10:
                main_issue = "High loan-to-income ratio"
                coach_advice = "The loan amount is too high relative to reported income. Reducing the loan amount or providing additional income proof could improve approval chances."
            else:
                main_issue = "Multiple risk factors"
                coach_advice = "The AI model detected multiple risk indicators in the financial profile. Consider improving credit score, reducing existing debt, and providing more comprehensive financial documentation."

        # ── Build response ────────────────────────────────────
        if status == "approved":
            main_issue = "None"

        response = {
            "status": status,
            "odds": odds,
            "main_issue": main_issue,
            "coach_advice": coach_advice,
            "bank_offers": bank_offers
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "status": "rejected",
            "odds": 0,
            "main_issue": f"Processing error: {str(e)}",
            "coach_advice": "An error occurred during analysis. Please verify all input fields contain valid numerical data and try again."
        }), 500


# ─── Legacy Route: Predict (kept for backward compatibility) ─
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ── Collect form values ─────────────────────────────
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
        total_income = applicant_income + coapplicant_income

        # ── Encode categorical values ───────────────────────
        gender_enc    = safe_encode(encoders["Gender"], gender)
        married_enc   = safe_encode(encoders["Married"], married)
        education_enc = safe_encode(encoders["Education"], education)
        self_emp_enc  = safe_encode(encoders["Self_Employed"], self_employed)

        # Dependents: "3+" → 3
        dependents = int(dependents_raw.replace("3+", "3"))

        # ── Feature array ────────────────────────────────────
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

        # Rule 1: No income → Reject
        if total_income <= 0:
            return render_template(
                "index.html",
                prediction="Loan Not Approved",
                approved=False,
                confidence=0,
                bank_offers=[],
                form_data=request.form
            )

        # Rule 2: Loan too high compared to income → Reject
        if loan_amount > total_income * 20:
           return render_template(
              "index.html",
               prediction="Loan Not Approved",
               approved=False,
               confidence=0,
               bank_offers=[],
               form_data=request.form
            )

        # ── Prediction ───────────────────────────────────────
        prediction_code = model.predict(features)[0]
        probability     = model.predict_proba(features)[0]

        bank_offers = []

        if prediction_code == 1:
            result   = "Loan Approved"
            approved = True
            conf     = round(probability[1] * 100, 1)

            # Bank suggestions based on confidence
            if conf >= 85:
                bank_offers = [
                    {"name": "HDFC Bank", "interest": "8.5% - 9.2%"},
                    {"name": "ICICI Bank", "interest": "8.7% - 9.5%"},
                    {"name": "State Bank of India", "interest": "8.6% - 9.1%"}
                ]
            elif conf >= 70:
                bank_offers = [
                    {"name": "Axis Bank", "interest": "9.0% - 10.5%"},
                    {"name": "Kotak Mahindra Bank", "interest": "9.2% - 10.8%"}
                ]
            else:
                bank_offers = [
                    {"name": "NBFC / Private Lenders", "interest": "10% - 14%"}
                ]

        else:
            result   = "Loan Not Approved"
            approved = False
            conf     = round(probability[0] * 100, 1)

        # ── Render page ──────────────────────────────────────
        return render_template(
            "index.html",
            prediction=result,
            approved=approved,
            confidence=conf,
            bank_offers=bank_offers,
            form_data=request.form
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}",
            approved=False,
            confidence=None,
            bank_offers=[],
            form_data=request.form
        )


# ─── Run App ─────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)