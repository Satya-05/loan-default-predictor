# 🏦 Loan Default Predictor

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-82.86%25-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning web application that predicts whether a loan applicant is likely to default, using the German Credit Dataset. Built with XGBoost, SHAP explainability, and deployed on Streamlit Cloud.

🔗 **Live Demo:** [Click here to try the app](https://loan-default-predictor-jfmjh9fxhuapa7wvhbjscz.streamlit.app/)

---

## 📌 Problem Statement

Banks and financial institutions face significant losses due to loan defaults. This project builds a machine learning pipeline to predict credit risk for loan applicants, helping institutions make data-driven lending decisions.

---

## 🚀 Features

- Predicts loan default risk with **82.86% accuracy**
- Compares 3 ML models: Logistic Regression, Random Forest, XGBoost
- **SHAP explainability** — shows why the model made each prediction
- Handles class imbalance using **SMOTE**
- Interactive web app with real-time predictions
- Clean EDA with visualizations

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 75.36% | 0.8264 |
| Random Forest | 81.43% | 0.9030 |
| **XGBoost** | **82.86%** | **0.9025** |

✅ XGBoost selected as the final model.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10
- **ML Libraries:** Scikit-learn, XGBoost, SHAP, Imbalanced-learn
- **Data:** German Credit Dataset (UCI / OpenML)
- **Web App:** Streamlit
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit Cloud

---

## 📁 Project Structure

```
loan-default-predictor/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_explainability.ipynb
├── app/
│   └── app.py
├── models/
│   ├── xgb_model.pkl
│   ├── rf_model.pkl
│   └── lr_model.pkl
├── data/
├── requirements.txt
└── README.md
```
---

## ⚙️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Satya-05/loan-default-predictor.git
cd loan-default-predictor

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
cd app
streamlit run app.py
```

---

## 🔍 Key Insights from EDA

- **70% good loans, 30% bad loans** — class imbalance handled with SMOTE
- **Checking account status** is the strongest predictor of default
- **Loan duration and credit amount** are highly correlated with risk
- Applicants with **no checking account** have the highest default rate

---

## 📈 SHAP Explainability

Top 5 features driving predictions:
1. `checking_status` — most important feature
2. `duration` — longer loans = higher risk
3. `credit_amount` — higher amounts = higher risk
4. `purpose` — loan purpose affects risk significantly
5. `residence_since` — stability indicator
