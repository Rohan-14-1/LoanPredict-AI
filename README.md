# LoanPredict-AI 💳

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat)

A full-stack AI-powered loan eligibility prediction web application. Enter your financial profile and get an instant decision — complete with approval odds, bank recommendations, EMI simulation, and an AI coach analysis.

---

## ✨ Features.

- **AI-Powered Prediction** — Random Forest Classifier trained on real loan data to predict approval or rejection with confidence scores.
- **Smart Analysis Dashboard** — Displays approval odds, primary risk factors, and personalized AI coach advice.
- **Bank Offer Engine** — Recommends suitable banks (SBI, HDFC, ICICI, Axis, etc.) with interest rates based on your credit profile.
- **Loan Simulator** — Interactive sliders to simulate how changing income, loan amount, duration, and interest rate affects your approval odds and EMI in real time.
- **Premium UI** — Multi-page glassmorphism interface with animated particle background, dark/light theme toggle, smooth page transitions, and 3D card tilt effects.
- **RESTful API** — Flask backend with a clean `/process_application` JSON endpoint for seamless frontend-backend communication.

---

## 🛠️ Tech Stack.

**Backend & Machine Learning**
- Python 3.8+
- Flask (Web Framework)
- Scikit-Learn (Random Forest Classifier)
- Pandas & NumPy (Data Processing)

**Frontend**
- HTML5 / CSS3 (Glassmorphism, CSS Variables, Flexbox/Grid)
- Vanilla JavaScript (Fetch API, Canvas API, Chart.js)
- Custom particle system, magnetic buttons, SPA-style page transitions

---

## 📁 Project Structure

```
LoanPredict-AI/
├── app.py                  # Flask server & API endpoints
├── train_model.py          # Model training script
├── loan_data.csv           # Training dataset
├── model.pkl               # Trained model artifact (generated)
├── requirements.txt        # Python dependencies
├── static/
│   ├── style.css           # Landing page styles
│   ├── script.js           # Landing page logic
│   ├── analysis.css        # Analysis page styles
│   ├── analysis.js         # Analysis logic & API calls
│   ├── dashboard.css       # Dashboard styles
│   ├── dashboard.js        # Dashboard charts & data
│   ├── simulator.css       # Simulator page styles
│   ├── simulator.js        # EMI & risk simulation engine
│   ├── navbar.css          # Global navbar & dark theme
│   ├── navbar.js           # Navbar, toast, transitions, tilt
│   └── particles.js        # Canvas particle background system
└── templates/
    ├── index.html          # Landing page
    ├── analysis.html       # Loan analysis form
    ├── dashboard.html      # Results dashboard
    ├── simulator.html      # Loan scenario simulator
    └── navbar.html         # Shared navbar fragment
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
Make sure you have **Python 3.8+** installed.

### 2. Clone the Repository
```bash
git clone https://github.com/Rohan-14-1/LoanPredict-AI.git
cd LoanPredict-AI
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
Run the training script to build the model and generate the `model.pkl` artifact:
```bash
python train_model.py
```

### 5. Start the Web Server
```bash
python app.py
```

### 6. Open the App
Navigate to: **http://127.0.0.1:5000**

---

## 🔌 API Reference

### `POST /process_application`
Accepts a JSON payload and returns a prediction result.

**Request Body**
```json
{
  "income": 85000,
  "loan_amount": 2500000,
  "loan_term": 360,
  "cibil_score": 750,
  "missed_payments": 0
}
```

**Response**
```json
{
  "status": "approved",
  "odds": 91.2,
  "main_issue": "None",
  "coach_advice": "Excellent financial profile...",
  "bank_offers": [
    { "name": "State Bank of India", "interest": "8.25%", "type": "Public Sector" }
  ]
}
```

---

## 📊 Model Details

| Attribute         | Detail                        |
|-------------------|-------------------------------|
| Algorithm         | Random Forest Classifier      |
| Training Split    | 80% train / 20% test          |
| Estimators        | 100 trees, max depth 6        |
| Key Features      | Income, Loan Amount, Credit History, Employment, Dependents |
| Output            | Binary (Approved / Rejected) + Probability Score |

---

## 📸 Pages Overview

| Page         | Route        | Description                                      |
|--------------|--------------|--------------------------------------------------|
| Landing      | `/`          | Hero page with features and CTA                  |
| Analysis     | `/analysis`  | 4-field diagnostic form with AI submission       |
| Dashboard    | `/dashboard` | Results with bank offers and coach feedback      |
| Simulator    | `/simulator` | Real-time EMI & risk simulation with Chart.js    |

---

## ⚠️ Disclaimer

This is an educational AI tool and does not constitute financial advice. Loan decisions depend on many additional factors evaluated by lending institutions. Always consult a certified financial advisor before applying for a loan.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
