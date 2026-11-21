from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "heart.csv"
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"

# These are the columns available in the supplied heart.csv file.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def main():
    df = pd.read_csv(DATA_PATH)

    # The original target has values 0, 1, 2, 3 and 4.
    # This project changes it to a binary target: 0 = no disease label,
    # greater than 0 = disease label, matching the supplied project PDF.
    df["target"] = (df["num"] > 0).astype(int)

    # ID and dataset/source are not health measurements, so they are not model inputs.
    X = df[FEATURES].copy()
    y = df["target"]

    # Keep boolean values consistent as strings for one-hot encoding.
    for col in ["fbs", "exang"]:
        X[col] = X[col].map({True: "True", False: "False"})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    model = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\\nMODEL EVALUATION")
    print("----------------")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, probabilities):.3f}")
    print("\\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": model,
        "features": FEATURES
    }, MODEL_PATH)
    print(f"\\nSaved trained model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
