from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def prepare_for_modeling(df, target="pneumonia_type"):
    categorical_features = ["gender", "severity", "comorbidity"]
    df = df.drop(columns=["patient_id"])

    top_features = [
        "procalcitonin",  # TOP FEATURE 1
        "pct_crp_ratio",  # TOP FEATURE 2
        "temperature_celsius",  # TOP FEATURE 3
        "wbc_count",  # TOP FEATURE 4
        "respiratory_distress",  # TOP FEATURE 5
        "symptom_count",
        "cough",
        "dyspnea",
        "fever",
        "fatigue",
        "sputum_production",
        "age",
        "respiratory_rate_bpm",
        "oxygen_saturation",
        "severity_num",
        "crp_level",
    ]

    df = df[categorical_features + top_features + [target]]

    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if target in numeric_features:
        numeric_features.remove(target)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        numeric_features,
        categorical_features,
    )