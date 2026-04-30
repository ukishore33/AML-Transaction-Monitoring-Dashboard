"""
AML Transaction Monitoring - Synthetic Data Generator
Author: Kishore U.
Description: Generates realistic synthetic transaction data with AML red flags
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"

np.random.seed(42)
random.seed(42)

COUNTRIES_HIGH_RISK = ["Iran", "Myanmar", "North Korea", "Syria", "Yemen", "Somalia"]
COUNTRIES_LOW_RISK  = ["India", "USA", "UK", "Germany", "Singapore", "Australia", "Japan"]
TRANSACTION_TYPES   = ["Wire Transfer", "Cash Deposit", "Cash Withdrawal", "SWIFT", "NEFT", "RTGS", "Crypto Exchange"]
CUSTOMER_SEGMENTS   = ["Retail", "Corporate", "SME", "HNI"]
INDUSTRIES          = ["IT Services", "Real Estate", "Import/Export", "Hospitality", "Finance", "Healthcare", "Retail Trade"]

def generate_transactions(n=2000):
    records = []
    start_date = datetime(2024, 1, 1)

    for i in range(n):
        is_suspicious = random.random() < 0.18  # ~18% suspicious

        cust_id   = f"CUST{random.randint(1000, 1500):04d}"
        txn_id    = f"TXN{i+1:06d}"
        segment   = random.choice(CUSTOMER_SEGMENTS)
        industry  = random.choice(INDUSTRIES)
        txn_type  = random.choice(TRANSACTION_TYPES)
        date      = start_date + timedelta(days=random.randint(0, 365))

        # Amount logic
        if is_suspicious:
            # Structuring: just below reporting thresholds
            if random.random() < 0.4:
                amount = round(random.uniform(4500, 9999), 2)
            else:
                amount = round(random.uniform(50000, 500000), 2)
        else:
            amount = round(np.random.lognormal(mean=9, sigma=1.5), 2)
            amount = min(amount, 200000)

        # Country risk
        if is_suspicious and random.random() < 0.55:
            country = random.choice(COUNTRIES_HIGH_RISK)
            country_risk = "High"
        else:
            country = random.choice(COUNTRIES_LOW_RISK)
            country_risk = "Low"

        # Velocity: transactions in last 24h for this customer
        velocity = random.randint(5, 20) if is_suspicious else random.randint(1, 5)

        # Round-number flag
        is_round = 1 if amount % 1000 == 0 else 0

        # Shell company flag
        shell_flag = 1 if (is_suspicious and random.random() < 0.4) else 0

        # PEP / Sanctions match
        pep_flag       = 1 if (is_suspicious and random.random() < 0.25) else 0
        sanctions_flag = 1 if (is_suspicious and random.random() < 0.2)  else 0

        # Multiple accounts same day
        multi_acc = 1 if (is_suspicious and random.random() < 0.35) else 0

        # Compute a simple manual risk score
        score = 0
        if country_risk == "High":   score += 35
        if amount > 10000:           score += 15
        if amount < 10000 and amount > 4000: score += 20  # structuring
        if velocity > 4:             score += 15
        if pep_flag:                 score += 20
        if sanctions_flag:           score += 25
        if shell_flag:               score += 20
        if multi_acc:                score += 15
        if is_round:                 score += 5
        score = min(score, 100)

        # SAR label (ground truth)
        sar = 1 if is_suspicious else 0

        records.append({
            "txn_id":           txn_id,
            "customer_id":      cust_id,
            "date":             date.strftime("%Y-%m-%d"),
            "txn_type":         txn_type,
            "amount_inr":       amount,
            "country":          country,
            "country_risk":     country_risk,
            "customer_segment": segment,
            "industry":         industry,
            "velocity_24h":     velocity,
            "is_round_number":  is_round,
            "shell_company_flag": shell_flag,
            "pep_flag":         pep_flag,
            "sanctions_flag":   sanctions_flag,
            "multi_account_flag": multi_acc,
            "risk_score":       score,
            "sar_label":        sar,
        })

    df = pd.DataFrame(records)
    df.to_csv(TRANSACTIONS_CSV, index=False)
    print(f"✅ Generated {n} transactions | Suspicious: {df['sar_label'].sum()} ({df['sar_label'].mean()*100:.1f}%)")
    print(f"   Saved dataset: {TRANSACTIONS_CSV}")
    return df

if __name__ == "__main__":
    generate_transactions()
