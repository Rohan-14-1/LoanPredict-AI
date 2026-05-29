# ============================================================
# app.py – Credit Risk Prediction | Flask Backend
# ============================================================

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import logging

app = Flask(__name__)

# ─── Setup logging ───────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Load the trained model artifact ─────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    
    model = artifact["model"]
    encoders = artifact["encoders"]
    FEATURES = artifact["features"]
    categorical_features = artifact.get("categorical_features", [])
    metrics = artifact.get("metrics", {})
    
    logger.info(f"✓ Model loaded successfully with {len(FEATURES)} features")
    logger.info(f"✓ Model Accuracy: {metrics.get('test_accuracy', 'N/A')}")
except Exception as e:
    logger.error(f"✗ Failed to load model: {e}")
    raise


# ─── Helper: safely encode a value with a LabelEncoder ──────
def safe_encode(encoder, value):
    """Safely encode a categorical value using the trained encoder."""
    try:
        classes = list(encoder.classes_)
        if value in classes:
            return classes.index(value)
        else:
            logger.warning(f"Value '{value}' not found in encoder classes. Using default (0)")
            return 0
    except Exception as e:
        logger.error(f"Error encoding value '{value}': {e}")
        return 0


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
    Credit Risk Prediction API
    
    Expected JSON payload:
    {
        "person_age": int,
        "person_income": float,
        "person_home_ownership": str (RENT/OWN/MORTGAGE),
        "person_emp_length": float,
        "loan_intent": str (PERSONAL/EDUCATION/MEDICAL/VENTURE/DEBTCONSOLIDATION/HOMEIMPROVEMENT),
        "loan_grade": str (A/B/C/D/E/F),
        "loan_amnt": float,
        "loan_int_rate": float,
        "loan_percent_income": float,
        "cb_person_default_on_file": str (Y/N),
        "cb_person_cred_hist_length": int
    }
    """
    try:
        data = request.get_json()
        
        logger.info(f"Processing application: {data}")
        
        # ── Extract and validate inputs ─────────────────────
        person_age = float(data.get("person_age", 30))
        person_income = float(data.get("person_income", 50000))
        person_home_ownership = data.get("person_home_ownership", "RENT")
        person_emp_length = float(data.get("person_emp_length", 5))
        loan_intent = data.get("loan_intent", "PERSONAL")
        loan_grade = data.get("loan_grade", "C")
        loan_amnt = float(data.get("loan_amnt", 10000))
        loan_int_rate = float(data.get("loan_int_rate", 10.0))
        loan_percent_income = float(data.get("loan_percent_income", 0.5))
        cb_person_default_on_file = data.get("cb_person_default_on_file", "N")
        cb_person_cred_hist_length = int(data.get("cb_person_cred_hist_length", 3))
        
        # ── Validate basic constraints ──────────────────────
        if person_income <= 0:
            return jsonify({
                "status": "rejected",
                "odds": 0,
                "risk_score": 100,
                "main_issue": "Zero or negative income",
                "coach_advice": "Income must be greater than zero. Please provide valid income information.",
                "confidence": 95
            }), 200
        
        if loan_amnt <= 0:
            return jsonify({
                "status": "rejected",
                "odds": 0,
                "risk_score": 100,
                "main_issue": "Invalid loan amount",
                "coach_advice": "Loan amount must be greater than zero.",
                "confidence": 95
            }), 200
        
        if loan_amnt > person_income * 10:
            return jsonify({
                "status": "rejected",
                "odds": 15,
                "risk_score": 85,
                "main_issue": "Loan amount too high relative to income",
                "coach_advice": f"Loan amount (₹{loan_amnt:,.0f}) is too high relative to income (₹{person_income:,.0f}). Consider reducing the loan amount.",
                "confidence": 90
            }), 200
        
        # ── Encode categorical features ────────────────────
        home_enc = safe_encode(encoders.get("person_home_ownership"), person_home_ownership)
        intent_enc = safe_encode(encoders.get("loan_intent"), loan_intent)
        grade_enc = safe_encode(encoders.get("loan_grade"), loan_grade)
        default_enc = safe_encode(encoders.get("cb_person_default_on_file"), cb_person_default_on_file)
        
        # ── Build feature array in correct order ────────────
        # Order must match FEATURES from training
        feature_values = {
            'person_age': person_age,
            'person_income': person_income,
            'person_home_ownership': home_enc,
            'person_emp_length': person_emp_length,
            'loan_intent': intent_enc,
            'loan_grade': grade_enc,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_status': 0,  # Placeholder for features alignment
            'loan_percent_income': loan_percent_income,
            'cb_person_default_on_file': default_enc,
            'cb_person_cred_hist_length': cb_person_cred_hist_length
        }
        
        # Build features array in FEATURES order
        features = []
        for feature_name in FEATURES:
            if feature_name in feature_values:
                features.append(feature_values[feature_name])
            else:
                logger.warning(f"Feature {feature_name} not provided, using default value")
                features.append(0)
        
        features = np.array([features])
        
        logger.info(f"Features shape: {features.shape}, FEATURES count: {len(FEATURES)}")
        
        # ── Make prediction ─────────────────────────────────
        try:
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            # probability[0] = class 0 (default/rejected)
            # probability[1] = class 1 (approved)
            odds = round(probabilities[1] * 100, 1) if len(probabilities) > 1 else 0
            risk_score = round((1 - probabilities[1]) * 100, 1) if len(probabilities) > 1 else 100
            
            logger.info(f"Prediction: {prediction}, Odds: {odds}%, Risk: {risk_score}%")
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({
                "status": "error",
                "odds": 0,
                "risk_score": 50,
                "main_issue": "Model prediction error",
                "coach_advice": "An error occurred during prediction. Please try again.",
                "confidence": 0
            }), 500
        
        # ── Determine status and provide guidance ──────────
        if prediction == 1:  # Approved
            status = "approved"
            
            if odds >= 90:
                main_issue = "None"
                coach_advice = f"Excellent profile! Strong income (₹{person_income:,.0f}), low debt ratio ({loan_percent_income:.1%}), and positive credit indicators. Highly recommended for approval with premium terms."
                bank_offers = [
                    {"name": "State Bank of India", "interest": "7.5% - 8.2%", "type": "Public Sector", "icon": "building-columns"},
                    {"name": "HDFC Bank", "interest": "8.0% - 8.5%", "type": "Private Sector", "icon": "landmark"},
                    {"name": "ICICI Bank", "interest": "8.2% - 8.8%", "type": "Private Sector", "icon": "building"},
                ]
            elif odds >= 75:
                main_issue = "None"
                coach_advice = f"Good financial profile with acceptable risk. Income level (₹{person_income:,.0f}) supports the requested loan. Standard rates recommended."
                bank_offers = [
                    {"name": "Axis Bank", "interest": "8.8% - 9.5%", "type": "Private Sector", "icon": "building"},
                    {"name": "Kotak Mahindra Bank", "interest": "9.0% - 9.7%", "type": "Private Sector", "icon": "landmark"},
                    {"name": "Punjab National Bank", "interest": "8.5% - 9.2%", "type": "Public Sector", "icon": "building-columns"}
                ]
            else:
                main_issue = "None"
                coach_advice = f"Borderline approval. Moderate risk indicators detected. Ensure repayment capacity of ₹{loan_amnt/12:,.0f}/month. Consider providing additional collateral."
                bank_offers = [
                    {"name": "Yes Bank", "interest": "10.0% - 11.0%", "type": "Private Sector", "icon": "building"},
                    {"name": "NBFC Partners", "interest": "10.5% - 12.0%", "type": "NBFC", "icon": "hand-holding-dollar"}
                ]
        
        else:  # Rejected
            status = "rejected"
            bank_offers = []
            
            if cb_person_default_on_file == "Y":
                main_issue = "Previous default on file"
                coach_advice = "The credit profile shows a previous default. Resolve outstanding dues and rebuild credit history before reapplying."
            elif loan_percent_income > 0.6:
                main_issue = "High debt-to-income ratio"
                coach_advice = f"Debt burden ({loan_percent_income:.1%}) is too high. Consider reducing loan amount or increasing income."
            elif person_emp_length < 2:
                main_issue = "Insufficient employment history"
                coach_advice = "Employment tenure is too short (less than 2 years). Reapply after gaining more employment stability."
            elif loan_grade in ['E', 'F']:
                main_issue = "Poor loan grade"
                coach_advice = "The loan has been graded as high-risk. Consider improving collateral or loan terms."
            else:
                main_issue = "Multiple risk factors"
                coach_advice = "The AI model detected multiple risk indicators. Improve credit profile, reduce existing debt, and provide comprehensive financial documentation."
        
        # ── Build response ──────────────────────────────────
        response = {
            "status": status,
            "odds": odds,
            "risk_score": risk_score,
            "main_issue": main_issue,
            "coach_advice": coach_advice,
            "bank_offers": bank_offers,
            "confidence": abs(max(probabilities) - 0.5) * 200 if len(probabilities) > 0 else 50  # Confidence measure
        }
        
        logger.info(f"Response: {response}")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Application processing error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "odds": 0,
            "risk_score": 50,
            "main_issue": "Processing error",
            "coach_advice": f"An error occurred: {str(e)}. Please verify all input fields contain valid data and try again.",
            "confidence": 0
        }), 500


# ─── Legacy Route: Predict (backward compatibility) ──────────
@app.route("/predict", methods=["POST"])
def predict():
    """Legacy predict route - redirects to process_application"""
    return process_application()


# ─── Run Application ────────────────────────────────────────
if __name__ == "__main__":
    # Use port 5001 instead of 5000 (macOS AirPlay uses 5000)
    app.run(debug=False, host="0.0.0.0", port=5001)


# ─── Run App ─────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)