def preprocess(df):
    df_processed = df.copy()

    key_symptoms = ["cough", "dyspnea", "fever", "fatigue", "sputum_production"]
    df_processed["symptom_count"] = df_processed[key_symptoms].sum(axis=1)

    df_processed["respiratory_distress"] = (
        df_processed["respiratory_rate_bpm"] / 20
        + (100 - df_processed["oxygen_saturation"]) / 5
        + df_processed["dyspnea"] * 2
    )

    severity_map = {"Mild": 1, "Moderate": 2, "Severe": 3, "Critical": 4}
    df_processed["severity_num"] = df_processed["severity"].map(severity_map)

    df_processed["pct_crp_ratio"] = df_processed["procalcitonin"] / (
        df_processed["crp_level"] + 1
    )

    return df_processed