"""
AML Transaction Monitoring - ML Model Training
Author: Kishore U.
Description: Trains Random Forest + Logistic Regression for SAR prediction
             Outputs model metrics and feature importances as JSON for dashboard
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
TRANSACTIONS_SCORED_CSV = DATA_DIR / "transactions_scored.csv"
MODEL_RESULTS_JSON = DATA_DIR / "model_results.json"

def train_model():
    df = pd.read_csv(TRANSACTIONS_CSV)

    # Feature engineering
    le = LabelEncoder()
    df["country_risk_enc"]     = le.fit_transform(df["country_risk"])
    df["segment_enc"]          = le.fit_transform(df["customer_segment"])
    df["txn_type_enc"]         = le.fit_transform(df["txn_type"])
    df["industry_enc"]         = le.fit_transform(df["industry"])
    df["amount_log"]           = np.log1p(df["amount_inr"])
    df["structuring_flag"]     = ((df["amount_inr"] >= 4500) & (df["amount_inr"] < 10000)).astype(int)
    df["high_value_flag"]      = (df["amount_inr"] > 100000).astype(int)
    df["combined_alert_score"] = (
        df["pep_flag"] + df["sanctions_flag"] +
        df["shell_company_flag"] + df["multi_account_flag"]
    )

    FEATURES = [
        "amount_log", "country_risk_enc", "velocity_24h",
        "is_round_number", "shell_company_flag", "pep_flag",
        "sanctions_flag", "multi_account_flag", "segment_enc",
        "txn_type_enc", "industry_enc", "structuring_flag",
        "high_value_flag", "combined_alert_score"
    ]

    X = df[FEATURES]
    y = df["sar_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    y_pred_rf   = rf.predict(X_test)
    y_proba_rf  = rf.predict_proba(X_test)[:, 1]

    # Logistic Regression
    lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr   = lr.predict(X_test)
    y_proba_lr  = lr.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "random_forest": {
            "accuracy":  round(float((y_pred_rf == y_test).mean()), 4),
            "precision": round(float(precision_score(y_test, y_pred_rf)), 4),
            "recall":    round(float(recall_score(y_test, y_pred_rf)), 4),
            "f1_score":  round(float(f1_score(y_test, y_pred_rf)), 4),
            "roc_auc":   round(float(roc_auc_score(y_test, y_proba_rf)), 4),
        },
        "logistic_regression": {
            "accuracy":  round(float((y_pred_lr == y_test).mean()), 4),
            "precision": round(float(precision_score(y_test, y_pred_lr)), 4),
            "recall":    round(float(recall_score(y_test, y_pred_lr)), 4),
            "f1_score":  round(float(f1_score(y_test, y_pred_lr)), 4),
            "roc_auc":   round(float(roc_auc_score(y_test, y_proba_lr)), 4),
        }
    }

    # Feature importances
    importances = sorted(
        zip(FEATURES, rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    feature_importance = [{"feature": f, "importance": round(float(v), 4)} for f, v in importances]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_rf).tolist()

    # Assign ML risk scores to full dataset
    df["ml_risk_score"] = (rf.predict_proba(X[FEATURES])[:, 1] * 100).round(1)
    df["ml_alert"]      = (df["ml_risk_score"] >= 50).astype(int)

    # Risk tiers
    def tier(score):
        if score >= 70: return "High"
        elif score >= 40: return "Medium"
        else: return "Low"
    df["risk_tier"] = df["ml_risk_score"].apply(tier)

    # Save enriched data
    df.to_csv(TRANSACTIONS_SCORED_CSV, index=False)

    # Save metrics JSON
    output = {
        "metrics": metrics,
        "feature_importance": feature_importance,
        "confusion_matrix": cm,
        "dataset_summary": {
            "total_transactions": int(len(df)),
            "total_suspicious":   int(df["sar_label"].sum()),
            "ml_alerts_raised":   int(df["ml_alert"].sum()),
            "high_risk_count":    int((df["risk_tier"] == "High").sum()),
            "medium_risk_count":  int((df["risk_tier"] == "Medium").sum()),
            "low_risk_count":     int((df["risk_tier"] == "Low").sum()),
            "high_risk_countries_txns": int((df["country_risk"] == "High").sum()),
            "total_amount_flagged": round(float(df[df["ml_alert"]==1]["amount_inr"].sum()), 2),
        }
    }

    with open(MODEL_RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print("✅ Model training complete!")
    print(f"   Random Forest  → AUC: {metrics['random_forest']['roc_auc']} | F1: {metrics['random_forest']['f1_score']}")
    print(f"   Logistic Reg.  → AUC: {metrics['logistic_regression']['roc_auc']} | F1: {metrics['logistic_regression']['f1_score']}")
    print(f"   ML Alerts raised: {output['dataset_summary']['ml_alerts_raised']}")

    return output

if __name__ == "__main__":
    train_model()
