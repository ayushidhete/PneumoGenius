import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def save_classification_report(y_test, y_pred, filename, title):
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    df_report = pd.DataFrame(report_dict).transpose()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=df_report.round(2).values,
        colLabels=df_report.columns,
        rowLabels=df_report.index,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


def build_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, multi_class="multinomial"
        ),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ), 
    }

    results = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring="accuracy"
        )
        cv_accuracy = cv_scores.mean()

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        results[name] = {
            "CV Accuracy": cv_accuracy,
            "Test Accuracy": accuracy,
            "Classification Report": class_report,
            "Confusion Matrix": conf_matrix,
            "Pipeline": pipeline,
        }

        report_filename = f"{name.replace(' ', '_').lower()}_classification_report.png"
        save_classification_report(
            y_test, y_pred, report_filename, title=f"{name} Classification Report"
        )

    return results