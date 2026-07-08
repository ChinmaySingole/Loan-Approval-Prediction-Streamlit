import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.card {
    background: #f8f9fa;
    padding: 1rem 1.2rem;
    border-radius: 14px;
    border: 1px solid #e6e6e6;
    margin-bottom: 1rem;
}
.small-text {
    color: #555;
    font-size: 0.92rem;
}
.good {
    color: #0f8a5f;
    font-weight: 600;
}
.bad {
    color: #c0392b;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    with open("best_model_name.pkl", "rb") as f:
        best_model_name = pickle.load(f)
    return model, scaler, label_encoders, feature_columns, best_model_name

model, scaler, label_encoders, feature_columns, best_model_name = load_artifacts()

APPROVED_LABEL = 0
REJECTED_LABEL = 1

def generate_reason_signals(data):
    strengths = []
    risks = []
    suggestions = []

    if data["person_income"] >= 80000:
        strengths.append("Strong annual income compared with typical applicants")
    elif data["person_income"] < 50000:
        risks.append("Lower annual income may reduce repayment confidence")
        suggestions.append("Increase declared stable income or add verified co-applicant income")

    if data["credit_score"] >= 700:
        strengths.append("Good credit score supports approval")
    elif data["credit_score"] < 600:
        risks.append("Low credit score is a major risk signal")
        suggestions.append("Improve credit score by repaying dues on time and reducing outstanding balances")

    if data["loan_interest_rate"] <= 10:
        strengths.append("Lower interest rate profile is favorable")
    elif data["loan_interest_rate"] > 13:
        risks.append("Higher interest rate indicates a riskier borrowing profile")
        suggestions.append("Try applying for a lower-risk loan profile or improve creditworthiness before reapplying")

    if data["loan_percentage"] <= 0.15:
        strengths.append("Loan percentage is within a safer range")
    elif data["loan_percentage"] > 0.25:
        risks.append("High loan percentage suggests higher repayment burden")
        suggestions.append("Reduce the requested loan amount or increase income to improve the ratio")

    if data["credit_history"] >= 5:
        strengths.append("Adequate credit history supports decision confidence")
    elif data["credit_history"] < 3:
        risks.append("Limited credit history may weaken trust in repayment behavior")
        suggestions.append("Build stronger credit history with consistent repayment records")

    if data["employee_experience"] >= 3:
        strengths.append("Work experience suggests better income stability")
    elif data["employee_experience"] == 0:
        risks.append("No work experience may indicate lower repayment stability")
        suggestions.append("Show stable employment history or additional income proof")

    if data["previous_loan"] == "Yes":
        strengths.append("Previous loan history may support lending familiarity")

    return strengths, risks, list(dict.fromkeys(suggestions))

st.title("💳 Loan Approval Prediction")
st.caption(f"Prediction app powered by: {best_model_name}")

st.markdown(
    ":blue-badge[Interactive Assessment] "
    ":green-badge[Applicant Summary] "
    ":orange-badge[Decision Signals] "
    ":violet-badge[Improvement Tips]"
)

st.write("Fill in the applicant details below and click **Predict Loan Status**.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=28)
    gender = st.selectbox("Gender", ["female", "male"])
    education = st.selectbox("Education", ["Associate", "Bachelor", "Doctorate", "High School", "Master"])
    person_income = st.number_input("Person Income", min_value=0.0, max_value=10000000.0, value=62000.0, step=1000.0)
    employee_experience = st.number_input("Employee Experience", min_value=0, max_value=60, value=4)
    home_onwership = st.selectbox("Home Onwership", ["MORTGAGE", "OTHER", "OWN", "RENT"])

with col2:
    loan_amount = st.number_input("Loan Amount", min_value=0.0, max_value=35000.0, value=11000.0, step=500.0)
    loan_intent = st.selectbox("Loan Intent", ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
    loan_interest_rate = st.number_input("Loan Interest Rate", min_value=0.0, max_value=20.0, value=10.8, step=0.1)
    loan_percentage = st.number_input("Loan Percentage", min_value=0.0, max_value=1.0, value=0.17, step=0.01)
    credit_history = st.number_input("Credit History", min_value=0, max_value=50, value=5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=668)
    previous_loan = st.selectbox("Previous Loan", ["No", "Yes"])

if st.button("Predict Loan Status", use_container_width=True):
    user_inputs = {
        "age": age,
        "gender": gender,
        "education": education,
        "person_income": person_income,
        "employee_experience": employee_experience,
        "home_onwership": home_onwership,
        "loan_amount": loan_amount,
        "loan_intent": loan_intent,
        "loan_interest_rate": loan_interest_rate,
        "loan_percentage": loan_percentage,
        "credit_history": credit_history,
        "credit_score": credit_score,
        "previous_loan": previous_loan
    }

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "education": education,
        "employee_experience": employee_experience,
        "home_onwership": home_onwership,
        "loan_amount": loan_amount,
        "loan_intent": loan_intent,
        "loan_interest_rate": loan_interest_rate,
        "loan_percentage": loan_percentage,
        "credit_history": credit_history,
        "credit_score": credit_score,
        "previous_loan": previous_loan,
        "person_income_log": np.log1p(person_income)
    }])

    categorical_cols = ["gender", "education", "home_onwership", "loan_intent", "previous_loan"]

    for col in categorical_cols:
        input_data[col] = label_encoders[col].transform(input_data[col])

    input_data = input_data[feature_columns]

    if "Logistic Regression" in best_model_name:
        input_processed = input_data.copy()
        numeric_features_only = input_processed.select_dtypes(include=np.number).columns.tolist()
        input_processed[numeric_features_only] = scaler.transform(input_processed[numeric_features_only])
    else:
        input_processed = input_data.copy()

    prediction = model.predict(input_processed)[0]
    probability = model.predict_proba(input_processed)[0]

    approval_prob = probability[APPROVED_LABEL]
    rejection_prob = probability[REJECTED_LABEL]

    strengths, risks, suggestions = generate_reason_signals(user_inputs)

    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Approval Probability", f"{approval_prob:.2%}")
    c2.metric("Rejection Probability", f"{rejection_prob:.2%}")
    c3.metric("Model Used", best_model_name)

    if prediction == APPROVED_LABEL:
        st.success("✅ Loan Status Prediction: Approved")
        st.badge("Low-to-Moderate Risk Profile", icon=":material/check_circle:", color="green")
    else:
        st.error("❌ Loan Status Prediction: Rejected")
        st.badge("Higher Risk Profile", icon=":material/warning:", color="red")

    with st.expander("Applicant Input Summary", expanded=True):
        summary_df = pd.DataFrame({
            "Field": [
                "Age", "Gender", "Education", "Person Income", "Employee Experience",
                "Home Onwership", "Loan Amount", "Loan Intent", "Loan Interest Rate",
                "Loan Percentage", "Credit History", "Credit Score", "Previous Loan"
            ],
            "Value": [
                age, gender, education, person_income, employee_experience,
                home_onwership, loan_amount, loan_intent, loan_interest_rate,
                loan_percentage, credit_history, credit_score, previous_loan
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with st.expander("Model Inputs Used for Prediction"):
        model_input_df = pd.DataFrame(input_data.T, columns=["Encoded / Final Value"])
        st.dataframe(model_input_df, use_container_width=True)

    with st.expander("Decision Signals", expanded=True):
        if strengths:
            st.markdown("### Positive Signals")
            for item in strengths:
                st.markdown(f"- <span class='good'>{item}</span>", unsafe_allow_html=True)

        if risks:
            st.markdown("### Risk Signals")
            for item in risks:
                st.markdown(f"- <span class='bad'>{item}</span>", unsafe_allow_html=True)

        if not strengths and not risks:
            st.info("No strong signals detected from the simple rule summary.")

    if prediction == REJECTED_LABEL:
        with st.expander("Suggestions to Improve Approval Chances", expanded=True):
            if suggestions:
                for item in suggestions:
                    st.markdown(f"- {item}")
            else:
                st.info("Maintain better credit behavior, reduce repayment burden, and improve income stability before reapplying.")

    st.caption("Note: Decision signals and suggestions are supportive insights based on entered values and project logic, not official bank lending advice.")