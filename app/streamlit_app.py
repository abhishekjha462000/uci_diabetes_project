import requests
import streamlit as st

st.set_page_config(page_title="Diabetes Readmission Predictor", layout="centered")

st.title("🏥 30-Day Diabetes Readmission Predictor")
st.write("Enter patient clinical details to estimate 30-day hospital readmission risk.")

API_URL = "http://127.0.0.1:8000/predict"

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.selectbox(
            "Age Group",
            ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
             "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
        )
        time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 4)
        num_lab_procedures = st.number_input("Lab Procedures", 0, 150, 45)
        num_procedures = st.number_input("Non-Lab Procedures", 0, 10, 1)
        num_medications = st.number_input("Number of Medications", 0, 100, 15)
        number_diagnoses = st.number_input("Number of Diagnoses", 1, 20, 7)

    with col2:
        number_outpatient = st.number_input("Prior Outpatient Visits", 0, 50, 0)
        number_emergency = st.number_input("Prior Emergency Visits", 0, 50, 0)
        number_inpatient = st.number_input("Prior Inpatient Visits", 0, 50, 1)
        max_glu_serum = st.selectbox("Max Glucose Serum", ["Not_Measured", "Norm", ">200", ">300"])
        A1Cresult = st.selectbox("A1C Result", ["Not_Measured", "Norm", ">7", ">8"])
        insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
        change = st.selectbox("Medication Changed?", ["No", "Ch"])
        diabetesMed = st.selectbox("Diabetes Medication Prescribed?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Readmission Risk", use_container_width=True)

if submitted:
    payload = {
        "race": race,
        "gender": gender,
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": A1Cresult,
        "insulin": insulin,
        "change": change,
        "diabetesMed": diabetesMed,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        prob = result["readmission_probability"]
        tier = result["risk_tier"]

        st.divider()
        st.metric("30-Day Readmission Probability", f"{prob * 100:.2f}%")

        if tier == "High":
            st.error(f"Risk Tier: **{tier}** — High likelihood of readmission within 30 days.")
        elif tier == "Medium":
            st.warning(f"Risk Tier: **{tier}** — Moderate readmission risk; consider follow-up.")
        else:
            st.success(f"Risk Tier: **{tier}** — Low likelihood of 30-day readmission.")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI. Make sure `uvicorn app.api:app --reload` is running.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")