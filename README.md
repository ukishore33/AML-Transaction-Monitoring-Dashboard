# 🔍 AML Transaction Monitoring Dashboard

> **End-to-end AML surveillance system** combining rule-based compliance logic with Machine Learning — built to replicate real-world financial crime detection workflows.

---

## 📌 Project Overview

This project simulates an AML (Anti-Money Laundering) transaction monitoring system used in banks and financial institutions. It generates synthetic transaction data with realistic red flags, trains an ML model for SAR (Suspicious Activity Report) prediction, and visualizes everything in an interactive compliance dashboard.

**Relevance:** Directly mirrors day-to-day work in AML/KYC compliance roles at organizations like Oracle, Deloitte, Tazapay, and major banks.

---

## 🏗️ Project Structure

```
aml-transaction-monitoring/
│
├── generate_data.py          # Synthetic transaction data generator (2,000 txns)
├── train_model.py            # ML model training — Random Forest + Logistic Regression
├── aml_dashboard.html        # Interactive compliance dashboard (Chart.js)
│
├── transactions.csv          # Raw synthetic dataset
├── transactions_scored.csv   # Enriched dataset with ML risk scores & tiers
├── model_results.json        # Model metrics, feature importances, confusion matrix
│
└── README.md
```

---

## 🚨 AML Red Flags Modelled

| Red Flag | Description |
|---|---|
| **Structuring** | Transactions just below ₹10,000 reporting threshold |
| **High-Risk Countries** | Iran, Myanmar, North Korea, Syria, Yemen, Somalia |
| **Transaction Velocity** | >4 transactions per customer in 24 hours |
| **PEP Match** | Politically Exposed Person flag |
| **Sanctions Hit** | OFAC/UN/EU sanctions list match |
| **Shell Company** | Opaque ownership structure flag |
| **Multi-Account Activity** | Same customer, multiple accounts, same day |
| **Round Number** | Suspiciously round transaction amounts |

---

## 🤖 ML Model

**Algorithm:** Random Forest Classifier (200 estimators, balanced class weights)  
**Comparison:** Logistic Regression (baseline)

### Features Used

| Feature | Importance |
|---|---|
| Transaction Velocity (24h) | 39.7% |
| Combined Alert Score | 19.6% |
| Country Risk Encoding | 13.4% |
| Amount (log-transformed) | 12.1% |
| High Value Flag | 4.8% |
| Multi-Account Flag | 2.6% |
| Shell Company Flag | 2.5% |
| PEP Flag | 1.4% |
| Structuring Flag | 1.2% |
| Sanctions Flag | 1.1% |

### Performance Metrics

| Metric | Random Forest | Logistic Regression |
|---|---|---|
| Accuracy | 1.00 | 1.00 |
| Precision | 1.00 | 1.00 |
| Recall | 1.00 | 1.00 |
| F1 Score | 1.00 | 1.00 |
| ROC-AUC | 1.00 | 1.00 |

> ⚠️ Perfect scores are expected on synthetic data where risk flags directly encode the label. In production, models operate on noisier signals with lower (but still meaningful) AUC ~0.80–0.90.

---

## 📊 Dashboard Features

- **Live Clock** with IST timezone
- **5 KPI Cards** — Total txns, ML alerts, high-risk country count, model accuracy, amount flagged
- **Monthly Volume vs Alerts** bar chart (Jan–Dec)
- **Alerts by Transaction Type** doughnut chart
- **Risk Tier Distribution** (High / Medium / Low)
- **Alerts by Country** horizontal bar chart
- **ML Model Performance** — metrics + confusion matrix
- **Feature Importance** bar visualization
- **AML Red-Flag Breakdown** — PEP, Sanctions, Shell Co., Structuring, etc.
- **SAR Alert Queue Table** — filterable by risk tier, country risk, transaction type, and search

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Generation | Python · Pandas · NumPy |
| ML Model | scikit-learn (Random Forest, Logistic Regression) |
| Feature Engineering | log-transform, encoding, derived flags |
| Visualization | Chart.js · Vanilla JS · HTML/CSS |
| AML Domain | KYC/KYB, Sanctions Screening, Transaction Monitoring, SAR logic |

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/aml-transaction-monitoring.git
cd aml-transaction-monitoring

# 2. Install dependencies
pip install pandas numpy scikit-learn

# 3. Generate synthetic data
python generate_data.py

# 4. Train the ML model
python train_model.py

# 5. Open the dashboard
open aml_dashboard.html   # macOS
# or just double-click aml_dashboard.html in Windows/Linux
```

---

## 👤 Author

**Kishore U.**  
AML/KYC Compliance Analyst · Data Analytics  
📧 UKISHORE33@GMAIL.COM · 📱 6303308133  
🔗 [LinkedIn](WWW.LINKEDIN.COM/IN/KISHORE-TECHIE/)

**Skills demonstrated:** AML | KYC/KYB | Sanctions Screening | Transaction Monitoring | Python | scikit-learn | SQL | Data Visualization | Financial Crime Compliance

---

## 📜 Disclaimer

All data in this project is **100% synthetic** and generated programmatically. No real customer, transaction, or financial data is used. This project is built solely for portfolio and educational demonstration purposes.
