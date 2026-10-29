"""Trains the risk model and saves it to models/risk_model.joblib"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
CATEGORICAL = ["road_type"]
NUMERIC = ["lat", "lon", "hour", "day_of_week", "month", "rainfall_mm",
           "visibility_km", "temperature_c", "junction", "traffic_volume", "speed_limit"]
TARGET = "accident"


def main():
    df = pd.read_csv(ROOT / "data" / "accidents.csv")
    X, y = df[CATEGORICAL + NUMERIC], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    model = Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=42)),
    ])
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
    print(classification_report(y_test, proba > 0.5))

    out = ROOT / "models" / "risk_model.joblib"
    joblib.dump(model, out)
    print("Model saved to", out)


if __name__ == "__main__":
    main()
