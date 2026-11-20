"""LoanLens - Flask app serving a scikit-learn decision tree for loan decisions."""
import math
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "decisiontree.pkl"))

# Which predicted class means "approved". Flip with APPROVED_CLASS=0 if your
# training labels are the other way round.
APPROVED_CLASS = int(os.getenv("APPROVED_CLASS", "1"))

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Only load pickle files you created yourself - unpickling can run code.
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
FEATURES = list(model.feature_names_in_)

# Label encodings (alphabetical, as sklearn's LabelEncoder produces them).
GENDER = {"female": 0, "male": 1}
EDUCATION = {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4}
HOME = {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3}
INTENT = {"DEBTCONSOLIDATION": 0, "EDUCATION": 1, "HOMEIMPROVEMENT": 2,
          "MEDICAL": 3, "PERSONAL": 4, "VENTURE": 5}
PREV_DEFAULT = {"No": 0, "Yes": 1}

# Options shown in the UI: (value, label), in a natural display order.
OPTIONS = {
    "gender": [("female", "Female"), ("male", "Male")],
    "education": [("High School", "High school"), ("Associate", "Associate"),
                  ("Bachelor", "Bachelor's"), ("Master", "Master's"), ("Doctorate", "Doctorate")],
    "home": [("RENT", "Rent"), ("MORTGAGE", "Mortgage"), ("OWN", "Own"), ("OTHER", "Other")],
    "intent": [("PERSONAL", "Personal"), ("DEBTCONSOLIDATION", "Debt consolidation"),
               ("EDUCATION", "Education"), ("HOMEIMPROVEMENT", "Home improvement"),
               ("MEDICAL", "Medical"), ("VENTURE", "Business venture")],
    "default": [("No", "No"), ("Yes", "Yes")],
}

# field -> (label, min, max, is_integer)
NUMERIC = {
    "person_age": ("Age", 18, 100, True),
    "person_income": ("Annual income", 1, 10_000_000, False),
    "person_emp_exp": ("Years of employment", 0, 80, True),
    "loan_amnt": ("Loan amount", 1, 10_000_000, False),
    "loan_int_rate": ("Interest rate", 0, 40, False),
    "cb_person_cred_hist_length": ("Credit history", 0, 80, True),
    "credit_score": ("Credit score", 300, 850, True),
}
CATEGORICAL = {
    "person_gender": GENDER,
    "person_education": EDUCATION,
    "person_home_ownership": HOME,
    "loan_intent": INTENT,
    "previous_loan_defaults_on_file": PREV_DEFAULT,
}
LABELS = {
    "person_age": "Age", "person_gender": "Gender", "person_education": "Education",
    "person_income": "Annual income", "person_emp_exp": "Employment experience",
    "person_home_ownership": "Home ownership", "loan_amnt": "Loan amount",
    "loan_intent": "Loan purpose", "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan as % of income", "cb_person_cred_hist_length": "Credit history",
    "credit_score": "Credit score", "previous_loan_defaults_on_file": "Previous default",
}

app = Flask(__name__)


def validate(payload):
    """Return (clean_values, errors)."""
    clean, errors = {}, {}
    for key, (label, lo, hi, is_int) in NUMERIC.items():
        try:
            val = float(payload.get(key))
            if not math.isfinite(val):
                raise ValueError
        except (TypeError, ValueError):
            errors[key] = f"Enter a number for {label.lower()}."
            continue
        if not lo <= val <= hi:
            errors[key] = f"{label} must be between {lo:,} and {hi:,}."
        elif is_int and val != int(val):
            errors[key] = f"{label} must be a whole number."
        else:
            clean[key] = int(val) if is_int else val
    for key, mapping in CATEGORICAL.items():
        if payload.get(key) not in mapping:
            errors[key] = f"Choose a valid option for {LABELS[key].lower()}."
        else:
            clean[key] = mapping[payload[key]]
    if "person_emp_exp" in clean and "person_age" in clean and clean["person_emp_exp"] > clean["person_age"]:
        errors["person_emp_exp"] = "Employment can't be longer than the applicant's age."
    return clean, errors


def top_factors(values, raw, n=5):
    """The model's most influential inputs, with this applicant's values."""
    imp = model.feature_importances_
    shown = {
        "person_income": f"${raw['person_income']:,.0f}",
        "loan_amnt": f"${raw['loan_amnt']:,.0f}",
        "loan_int_rate": f"{raw['loan_int_rate']:.2f}%",
        "loan_percent_income": f"{values['loan_percent_income'] * 100:.0f}%",
        "credit_score": str(raw["credit_score"]),
        "previous_loan_defaults_on_file": raw["previous_loan_defaults_on_file"],
        "person_age": str(raw["person_age"]),
        "person_emp_exp": f"{raw['person_emp_exp']} yrs",
        "cb_person_cred_hist_length": f"{raw['cb_person_cred_hist_length']} yrs",
    }
    out = []
    for i in np.argsort(imp)[::-1]:
        name = FEATURES[i]
        value = shown.get(name) or str(raw.get(name, ""))
        out.append({"label": LABELS[name], "value": value, "weight": round(float(imp[i]) * 100, 1)})
        if len(out) == n:
            break
    return out


@app.route("/")
def index():
    return render_template("index.html", options=OPTIONS)


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    clean, errors = validate(payload)
    if errors:
        return jsonify(errors=errors), 422

    income, amount = clean["person_income"], clean["loan_amnt"]
    row = dict(clean)
    row["loan_percent_income"] = round(amount / income, 2)
    x = np.array([[row[name] for name in FEATURES]], dtype=float)
    pred = int(model.predict(x)[0])
    approved = pred == APPROVED_CLASS

    return jsonify(
        approved=approved,
        loan_percent_income=row["loan_percent_income"],
        factors=top_factors(row, payload),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
