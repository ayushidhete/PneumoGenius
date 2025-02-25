import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)


def generate_pneumonia_dataset(num_patients=100):
    genders = ["Male", "Female"]
    pneumonia_types = ["Bacterial", "Viral", "Fungal", "Aspiration"]
    severity_levels = ["Mild", "Moderate", "Severe", "Critical"]
    comorbidities = [
        "None",
        "Diabetes",
        "Hypertension",
        "COPD",
        "Asthma",
        "Heart Disease",
    ]

    patient_ids = [f"P{1000 + i}" for i in range(num_patients)]

    data = {
        "patient_id": patient_ids,
        "age": np.random.normal(65, 15, num_patients).astype(int).clip(18, 95),
        "gender": [random.choice(genders) for _ in range(num_patients)],
        "pneumonia_type": [random.choice(pneumonia_types) for _ in range(num_patients)],
        "severity": [random.choice(severity_levels) for _ in range(num_patients)],
        "comorbidity": [random.choice(comorbidities) for _ in range(num_patients)],
    }

    severity_map = {"Mild": 0, "Moderate": 1, "Severe": 2, "Critical": 3}
    severity_indices = [severity_map[s] for s in data["severity"]]

    base_temp = np.random.normal(37.0, 0.3, num_patients)
    fever_addition = np.array([0.5, 1.0, 1.5, 2.0])[severity_indices]
    data["temperature_celsius"] = (base_temp + fever_addition).round(1).clip(36.0, 41.0)

    base_rr = np.random.normal(14, 2, num_patients)
    rr_addition = np.array([2, 4, 8, 14])[severity_indices]
    data["respiratory_rate_bpm"] = (base_rr + rr_addition).astype(int).clip(12, 40)

    base_o2 = np.random.normal(98, 1, num_patients)
    o2_reduction = np.array([2, 5, 10, 20])[severity_indices]
    data["oxygen_saturation"] = (base_o2 - o2_reduction).round(1).clip(75, 100)

    symptom_probs = {
        "cough": [0.8, 0.9, 0.95, 0.98],
        "dyspnea": [0.4, 0.7, 0.9, 0.98],
        "fever": [0.6, 0.8, 0.9, 0.95],
        "fatigue": [0.7, 0.8, 0.9, 0.95],
        "sputum_production": [0.6, 0.7, 0.8, 0.9],
    }

    for symptom, probs in symptom_probs.items():
        thresholds = np.array(probs)[severity_indices]
        data[symptom] = (np.random.random(num_patients) < thresholds).astype(int)

    base_wbc = np.random.normal(7, 1, num_patients)
    wbc_addition = np.array([2, 4, 7, 12])[severity_indices]
    data["wbc_count"] = (base_wbc + wbc_addition).round(1).clip(4, 30)

    base_crp = np.random.normal(5, 2, num_patients)
    crp_addition = np.array([5, 20, 80, 200])[severity_indices]
    data["crp_level"] = (base_crp + crp_addition).round(1).clip(0, 350)

    base_pct = np.random.normal(0.1, 0.05, num_patients)
    pct_addition = np.zeros(num_patients)

    for i in range(num_patients):
        if data["pneumonia_type"][i] == "Bacterial":
            pct_addition[i] = np.array([0.2, 0.5, 5, 20])[severity_indices[i]]
        else:
            pct_addition[i] = np.array([0.05, 0.1, 0.3, 1])[severity_indices[i]]

    data["procalcitonin"] = (base_pct + pct_addition).round(2).clip(0, 100)

    mortality_probs = np.array([0.01, 0.05, 0.15, 0.3])[severity_indices]
    data["deceased"] = (np.random.random(num_patients) < mortality_probs).astype(int)

    df = pd.DataFrame(data)

    return df


pneumonia_df = generate_pneumonia_dataset(500)
pneumonia_df.to_csv("pneumonia.csv", index=False)

# # Summary statistics
# print("\nDataset Summary:")
# print(f"Number of patients: {len(pneumonia_df)}")
# print(f"Pneumonia types distribution:\n{pneumonia_df['pneumonia_type'].value_counts()}")
# print(f"Severity distribution:\n{pneumonia_df['severity'].value_counts()}")
# print(f"Average age: {pneumonia_df['age'].mean():.1f} years")
# print(f"Average oxygen saturation: {pneumonia_df['oxygen_saturation'].mean():.1f}%")
# print(f"Mortality rate: {pneumonia_df['deceased'].mean() * 100:.1f}%")