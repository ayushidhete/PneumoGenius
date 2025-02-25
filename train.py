import pandas as pd
from explore import explore_data
from modelling import prepare_for_modeling
from models import build_and_evaluate_models
import joblib
from preprocess import preprocess

pneumonia_df = pd.read_csv("data/pneumonia.csv")
target_distribution = explore_data(pneumonia_df)
pneumonia_df_processed = preprocess(pneumonia_df)

(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    numeric_features,
    categorical_features,
) = prepare_for_modeling(pneumonia_df_processed)


model_results = build_and_evaluate_models(
    X_train, X_test, y_train, y_test, preprocessor
)

best_model_name = max(model_results, key=lambda k: model_results[k]["Test Accuracy"])
best_model = model_results[best_model_name]["Pipeline"]
print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {model_results[best_model_name]['Test Accuracy']:.4f}")


def predict_pneumonia_type(patient_data, model=best_model):
    patient_df = pd.DataFrame([patient_data])
    patient_df_processed = preprocess(patient_df)
    predicted_type = model.predict(patient_df_processed)[0]
    probabilities = model.predict_proba(patient_df_processed)[0]
    pneumonia_types = model.classes_
    prob_dict = {
        pneumonia_types[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities)
    }
    return predicted_type, prob_dict


example_patient = {
    "patient_id": "P9999",
    "age": 65,
    "gender": "Male",
    "severity": "Moderate",
    "comorbidity": "Diabetes",
    "temperature_celsius": 38.5,
    "respiratory_rate_bpm": 22,
    "oxygen_saturation": 93.0,
    "cough": 1,
    "dyspnea": 1,
    "fever": 1,
    "fatigue": 1,
    "sputum_production": 1,
    "wbc_count": 12.5,
    "crp_level": 45.0,
    "procalcitonin": 0.75,
}

predicted_type, type_probabilities = predict_pneumonia_type(example_patient)
print(f"\nPredicted pneumonia type for example patient: {predicted_type}")
print("Probability for each type:")
for ptype, prob in type_probabilities.items():
    print(f"  {ptype}: {prob}%")


joblib.dump(best_model, "pred_model.pkl")
print("\nBest model saved as 'pred_model.pkl'")

# # Generate a summary report of the key features and symptoms
# print("\n===== PNEUMONIA CLASSIFICATION SUMMARY =====")
# print("Key Symptoms Used in Model:")
# for i, symptom in enumerate(
#     ["Cough", "Dyspnea", "Fever", "Fatigue", "Sputum Production"]
# ):
#     print(f"{i+1}. {symptom}")

# print("\nTop 5 Predictive Features:")
# print("1. Procalcitonin Level - Higher in bacterial pneumonia")
# print("2. PCT/CRP Ratio - Ratio of procalcitonin to C-reactive protein")
# print("3. Temperature - Fever patterns differ by pneumonia type")
# print("4. WBC Count - White blood cell count varies by infection type")
# print("5. Respiratory Distress Score - Composite measure of respiratory function")