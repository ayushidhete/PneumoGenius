import matplotlib.pyplot as plt
import seaborn as sns


def explore_data(df, target_column="pneumonia_type"):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=target_column, data=df)
    plt.title("Distribution of Pneumonia Types")
    plt.savefig("pneumonia_type_distribution.png")

    categorical_features = ["gender", "severity", "comorbidity"]
    for feature in categorical_features:
        plt.figure(figsize=(12, 6))
        sns.countplot(x=feature, hue=target_column, data=df)
        plt.title(f"{feature} vs {target_column}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{feature}_vs_{target_column}.png")

    key_numerics = [
        "temperature_celsius",
        "wbc_count",
        "procalcitonin",
        "crp_level",
        "respiratory_rate_bpm",
    ]
    for feature in key_numerics:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=target_column, y=feature, data=df)
        plt.title(f"{feature} by {target_column}")
        plt.tight_layout()
        plt.savefig(f"{feature}_by_{target_column}.png")

    key_symptoms = ["cough", "dyspnea", "fever", "fatigue", "sputum_production"]
    symptom_data = df.groupby([target_column])[key_symptoms].mean()

    plt.figure(figsize=(12, 8))
    symptom_data.T.plot(kind="bar")
    plt.title("Symptom Prevalence by Pneumonia Type")
    plt.ylabel("Prevalence (%)")
    plt.xticks(rotation=45)
    plt.legend(title="Pneumonia Type")
    plt.tight_layout()
    plt.savefig("symptom_prevalence.png")

    return df[target_column].value_counts()