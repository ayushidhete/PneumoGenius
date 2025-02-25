import streamlit as st
import pandas as pd
from groq import Groq
# import pickle as pl
import joblib as joblib
# import pickle as pl
import os
from dotenv import load_dotenv
from preprocess import preprocess
import numpy as np

load_dotenv()

st.set_page_config(page_title="PneumoGenius", page_icon="🩺", layout="wide")

model = joblib.load("model/pred_model.pkl")
# with open("model/pred_model.pkl", "rb") as f:
#     model = pl.load(f)

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)


def predict_pneumonia_type(patient_data, model=model):
    patient_df = pd.DataFrame([patient_data])
    patient_df_processed = preprocess(patient_df)
    print(patient_df_processed.dtypes)
    predicted_type = model.predict(patient_df_processed)[0]
    return predicted_type, patient_data


st.sidebar.title("Navigation")
report_btn = st.sidebar.button("📄 Report")
stats_btn = st.sidebar.button("📊 Model Stats")
image_btn = st.sidebar.button("🖼️ Images")

if report_btn:
    st.session_state["page"] = "Report"
elif stats_btn:
    st.session_state["page"] = "Model Stats"
elif image_btn:
    st.session_state["page"] = "Images"

if "page" not in st.session_state:
    st.session_state["page"] = "Report"

if st.session_state["page"] == "Report":
    st.title("🩺 PneumoGenius - Your Own Diagnostic App")
    st.write(
        "### Enter patient details to predict the pneumonia type and get detailed analysis."
    )

    # Create three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📌 Personal Information")
        age = st.number_input("Age", min_value=0, max_value=120, value=65)
        gender = st.selectbox("Gender", ["Male", "Female"])
        severity = st.selectbox("Severity", ["Mild", "Moderate", "Severe"])
        comorbidity = st.selectbox(
            "Comorbidity",
            ["None", "Diabetes", "Hypertension", "Heart Disease", "Asthma", "COPD"],
        )

    with col2:
        st.subheader("🌡 Vitals")
        temperature = st.number_input(
            "Temperature (°C)", min_value=35.0, max_value=42.0, value=38.5
        )
        respiratory_rate = st.number_input(
            "Respiratory Rate (bpm)", min_value=10, max_value=40, value=22
        )
        oxygen_saturation = st.number_input(
            "Oxygen Saturation (%)", min_value=70.0, max_value=100.0, value=93.0
        )
        wbc_count = st.number_input(
            "WBC Count (10^9/L)", min_value=2.0, max_value=20.0, value=12.5
        )

    with col3:
        st.subheader("🤧 Symptoms & Lab Tests")
        cough = st.checkbox("Cough")
        dyspnea = st.checkbox("Dyspnea")
        fever = st.checkbox("Fever")
        fatigue = st.checkbox("Fatigue")
        sputum_production = st.checkbox("Sputum Production")
        crp_level = st.number_input(
            "CRP Level (mg/L)", min_value=0.0, max_value=200.0, value=45.0
        )
        procalcitonin = st.number_input(
            "Procalcitonin (ng/mL)", min_value=0.0, max_value=10.0, value=0.75
        )

    if st.button("Get Diagnosis!"):
        patient_data = {
            "age": age,
            "gender": gender,
            "severity": severity,
            "comorbidity": comorbidity,
            "temperature_celsius": temperature,
            "respiratory_rate_bpm": respiratory_rate,
            "oxygen_saturation": oxygen_saturation,
            "cough": int(cough),
            "dyspnea": int(dyspnea),
            "fever": int(fever),
            "fatigue": int(fatigue),
            "sputum_production": int(sputum_production),
            "wbc_count": wbc_count,
            "crp_level": crp_level,
            "procalcitonin": procalcitonin,
        }

        predicted_type, _ = predict_pneumonia_type(patient_data)

        output_data = []

        output_data.append("📌 Patient Personal Information")
        output_data.append(f"**Age (shows age of the person):** {patient_data['age']}")
        output_data.append(
            f"**Gender (shows patient's gender):** {patient_data['gender']}"
        )
        output_data.append(
            f"**Severity (indicates pneumonia severity):** {patient_data['severity']}"
        )
        output_data.append(
            f"**Comorbidity (shows any existing health condition):** {patient_data['comorbidity']}"
        )

        output_data.append("🌡 Vital Signs")
        output_data.append(
            f"**Temperature (°C) (body temperature of the patient):** {patient_data['temperature_celsius']}"
        )
        output_data.append(
            f"**Respiratory Rate (bpm) (breathing rate per minute):** {patient_data['respiratory_rate_bpm']}"
        )
        output_data.append(
            f"**Oxygen Saturation (%) (oxygen level in blood):** {patient_data['oxygen_saturation']}"
        )
        output_data.append(
            f"**WBC Count (10^9/L) (white blood cell count):** {patient_data['wbc_count']}"
        )

        output_data.append("🤧 Symptoms & Lab Tests")
        output_data.append(
            f"**Cough (whether the patient has a cough or not):** {'yes' if patient_data['cough'] else 'no'}"
        )
        output_data.append(
            f"**Dyspnea (difficulty in breathing):** {'yes' if patient_data['dyspnea'] else 'no'}"
        )
        output_data.append(
            f"**Fever (high body temperature):** {'yes' if patient_data['fever'] else 'no'}"
        )
        output_data.append(
            f"**Fatigue (feeling of tiredness or weakness):** {'yes' if patient_data['fatigue'] else 'no'}"
        )
        output_data.append(
            f"**Sputum Production (coughing up mucus):** {'yes' if patient_data['sputum_production'] else 'no'}"
        )
        output_data.append(
            f"**CRP Level (mg/L) (C-reactive protein level, indicating inflammation):** {patient_data['crp_level']}"
        )
        output_data.append(
            f"**Procalcitonin (ng/mL) (biomarker for bacterial infection):** {patient_data['procalcitonin']}"
        )

        output_data.append("Results")
        output_data.append(
            f"Predicted Pneumonia Type Result (determined pneumonia classification): **{predicted_type}**"
        )

        if "all_patient_data" not in st.session_state:
            st.session_state["all_patient_data"] = []

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are helpful Diagnostic Assistant provide a detailed report based on the information provided and also provide cure information,",
                },
                {
                    "role": "user",
                    "content": "".join(output_data),
                },
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )

        final_output = chat_completion.choices[0].message.content
        st.session_state["all_patient_data"].append([final_output])

        st.session_state["page"] = "Results"
        st.rerun()

elif st.session_state["page"] == "Model Stats":
    st.title("📊 Model Stats")

    img_folder = "img"

    if os.path.exists(img_folder) and os.path.isdir(img_folder):
        img_files = [
            f
            for f in os.listdir(img_folder)
            if f.endswith((".png", ".jpg", ".jpeg", ".gif"))
        ]

        if img_files:
            for img in img_files:
                img_path = os.path.join(img_folder, img)
                st.image(img_path, caption=img, use_container_width=True)
        else:
            st.warning("No images found in the 'img' folder.")
    else:
        st.error("Image folder not found.")

    if st.button("🔙 Go Back"):
        st.session_state["page"] = "Report"
        st.rerun()


elif st.session_state["page"] == "Results":
    st.title("📝 Diagnostic Results")
    if "all_patient_data" in st.session_state:
        for data in st.session_state["all_patient_data"]:
            for item in data:
                st.write(item)
    if st.button("🔙 Go Back"):
        st.session_state["page"] = "Report"
        st.rerun()