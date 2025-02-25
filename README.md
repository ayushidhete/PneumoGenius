# 🩺 PneumoGenius - Your Own Diagnostic App

A comprehensive machine learning solution that predicts pneumonia type , then generates detailed diagnosis reports using LLM integration.

## Project Overview

This project develops a two-stage predictive system for pneumonia patients:

1. **Prediction Engine**: Uses machine learning to predict both pneumonia type (Bacterial, Viral, Fungal, or Aspiration) and expected length of hospital stay
2. **Diagnostic Report Generation**: Leverages Large Language Models (via Groq) to generate comprehensive, human-readable diagnostic reports based on patient data and predictions

This system can help healthcare providers with diagnosis, treatment planning, resource allocation, and patient communication.

## Features

- Synthetic dataset generation of realistic pneumonia patient data
- Comprehensive exploratory data analysis (EDA)
- Feature engineering to create clinically relevant predictors
- Multiple regression models for length of stay prediction
- Classification model for pneumonia type prediction
- LLM integration for detailed diagnostic report generation
- Feature importance analysis
- Ready-to-use prediction and reporting functions

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- groq (for LLM integration)

## Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/ayushidhete/PneumoGenius.git
cd PneumoGenius
pip install -r requirements.txt
```

## Project Structure

```
PneumoGenius/              # Pneumonia Diagnosis and Prediction System
│
├── app.py                 # Streamlit app for user interaction and diagnosis
├── dataset.py             # Generates synthetic pneumonia dataset
├── train.py               # Trains models, evaluates performance, and creates visualizations
├── modelling.py           # Defines ML models for pneumonia classification
├── explore.py             # Exploratory data analysis and feature insights
├── models.py              # Stores model-related utilities and functions
├── preprocess.py          # Data preprocessing and feature engineering functions
│
├── model/                 # Saved trained models
│ ├── pred.pkl             # Pneumonia classification model
│
├── data/                  # Dataset files
│ ├── pneumonia.csv        # Pneumonia dataset
│
├── img/                   # Generated charts and graphs
│ ├── feature.png          # Visualization of important features
│ ├── ...                  # Additional graphs and images
│
├── README.md              # Project documentation and instructions
├── .gitignore             # Files to ignore in version control
├── .env                   # Environment variables (e.g., API keys)
└── requirements.txt       # List of dependencies and libraries
```

## Workflow

### 1. Generate the Dataset

First, run the dataset generation script to create a synthetic pneumonia patient dataset:

```bash
python dataset.py
```

This will:

- Generate a comprehensive dataset of pneumonia patients with realistic clinical features
- Save the dataset as `data/pneumonia_dataset.csv`

### 2. Train the Models

Next, train the prediction models using the generated dataset:

```bash
python train.py
```

This will:

- Load the dataset from `data/pneumonia_dataset.csv`
- Perform exploratory data analysis
- Engineer relevant features
- Train and evaluate multiple models for pneumonia type prediction and length of stay prediction
- Generate various visualization files in the `img/` directory, including:
  - Feature importance bar graphs
  - Confusion matrices
  - ROC curves
  - Actual vs. predicted comparison charts
  - Classification reports
- Save the best performing models to the `models/` directory

### 3. Set Up Environment Variables

Before running the app, set up your GROQ API key in the .env file:

```bash
# Create or edit your .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

### 4. Launch the Streamlit App

Finally, run the Streamlit app to interact with the prediction system:

```bash
streamlit run app.py
```

This will:

- Start a local web server
- Open the PneumoGenius app in your default web browser
- Allow you to input patient data and receive pneumonia type predictions, length of stay estimates, and comprehensive diagnostic reports

### 5. Using the Prediction and Diagnosis System

After the models are trained, you can also use them programmatically:

```python
from predict import predict_pneumonia_type, generate_diagnosis_report

# Patient data
patient_data = {
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

predicted_type = predict_pneumonia_type(patient_data)
print(f"Predicted pneumonia type: {predicted_type}")

diagnosis_report = generate_diagnosis_report(patient_data, predicted_type)
print(diagnosis_report)
```

## Data Description

The model uses the following features:

### Required Input Features

- **Demographics**:

  - `age`: Patient's age in years
  - `gender`: Patient's gender ("Male" or "Female")

- **Clinical Assessment**:

  - `severity`: Severity level of pneumonia ("Mild", "Moderate", "Severe", or "Critical")
  - `comorbidity`: Existing health condition ("None", "Diabetes", "Hypertension", "COPD", "Asthma", etc.)

- **Vital Signs**:

  - `temperature_celsius`: Body temperature in Celsius
  - `respiratory_rate_bpm`: Breathing rate in breaths per minute
  - `oxygen_saturation`: Blood oxygen level as percentage

- **Symptoms** (binary values, 1=present, 0=absent):

  - `cough`: Presence of cough
  - `dyspnea`: Difficulty breathing
  - `fever`: Subjective feeling of fever
  - `fatigue`: Feeling of tiredness
  - `sputum_production`: Production of mucus/phlegm

- **Laboratory Values**:
  - `wbc_count`: White blood cell count (×10⁹/L)
  - `crp_level`: C-reactive protein level (mg/L)
  - `procalcitonin`: Procalcitonin level (ng/mL)

## Generated Visualizations

During the training process (`train.py`), the following visualizations are generated:

1. **Feature Importance**: Bar charts showing the most influential features for both pneumonia type and length of stay prediction
2. **Confusion Matrix**: Visual representation of classification performance for pneumonia type prediction
3. **ROC Curves**: For evaluating classification performance
4. **Severity vs. Length of Stay**: Box plots showing relationship between pneumonia severity and hospital stay duration
5. **Age and Severity vs. Length of Stay**: Scatter plots showing how age and severity interact to affect hospital stay
6. **Correlation Matrix**: Heatmap showing relationships between different clinical features
7. **Classification Report**: Detailed metrics on classification performance
8. **Actual vs. Predicted Plots**: For assessing regression model performance

These visualizations are saved to the `visualizations/` directory and can be used for model evaluation, clinical insights, and presentations.

## LLM Integration for Diagnostic Reports

The system uses Groq's LLM API to generate detailed diagnostic reports. The process works as follows:

1. Patient data is collected and formatted
2. ML models predict pneumonia type and length of stay
3. All data is formatted into a structured prompt with sections including:
   - Patient Personal Information
   - Vital Signs
   - Symptoms & Lab Tests
   - Prediction Results
4. The formatted data is sent to Groq's LLM API
5. The LLM generates a comprehensive, natural language diagnostic report that includes:
   - Analysis of patient symptoms and vital signs
   - Interpretation of laboratory findings
   - Detailed explanation of the predicted pneumonia type
   - Treatment considerations based on the diagnosis
   - Prognosis information including expected length of stay

This integration creates human-readable reports that can be used for clinical decision support and patient communication.

## Model Performance

The project evaluates multiple models including:

- For pneumonia type prediction: Classification models (Random Forest, Gradient Boosting, etc.)

Performance metrics used:

- For classification: Accuracy, Precision, Recall, F1-Score
- For regression: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R² Score

## Limitations

- The models are trained on synthetic data and should be validated with real clinical data before use in healthcare settings
- LLM-generated reports should be reviewed by qualified healthcare professionals
- Predictions are estimates and should be considered alongside clinical judgment
- The system requires all input features to be present for accurate predictions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses a synthetic data generator to create realistic pneumonia patient profiles
- The approach leverages both traditional ML and modern LLM technologies
- The system architecture can be adapted to other clinical prediction and reporting tasks