# 🏦 Bank Customer Churn Prediction

LIVE: https://rajeevkumar75-bank-customer-churn-prediction-app-ict6p9.streamlit.app/

This project focuses on predicting customer churn for a retail bank using a complete **end-to-end machine learning workflow**.  
The main goal is to identify customers who are likely to leave the bank early so that retention actions can be taken in time.

The project covers **data analysis, feature engineering, model training, tuning, and deployment**, with a strong focus on practical and business-oriented decision making.

---

## 🔍 Problem Statement
Customer churn directly impacts a bank’s revenue and long-term growth.  
Instead of reacting after customers leave, banks need a system that can **flag high-risk customers in advance**.

This project builds such a system using historical customer data and machine learning.

---

## 📊 Dataset Information
- **Source:** Kaggle (Bank Customer Churn Dataset)
- **Total Records:** 10,000
- **Features:** 12
- **Target Variable:** `churn` (1 = churned, 0 = retained)
- **Churn Percentage:** ~20%

### Main Features
`credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary`

---

## 📈 Key Findings from Exploratory Data Analysis

Some important patterns observed from the data:

- **Age is the most influential factor**
  - Customers between **46–65 years** show the highest churn
  - Younger customers (around 30–35) are more loyal
- **Female customers churn slightly more than male customers**
- **Germany** has a much higher churn rate (~32%) compared to France and Spain (~16%)
- Customers who churn often have **higher account balances**
- High-value customers may churn due to **better alternatives or higher service expectations**

📌 **Business takeaway:**  
Middle-aged and high-balance customers should be targeted with personalized retention strategies.

---

## 🛠 Feature Engineering & Preprocessing
To improve model performance, several meaningful features were created:

- `balance_salary_ratio`
- `tenure_age_ratio`
- `balance_per_product`
- `is_senior`
- `zero_balance_flag`

Additional preprocessing steps:
- Multicollinearity handled using **VIF**
- Class imbalance handled with **SMOTE**
- Categorical variables encoded properly
- Numerical features normalized
- Outliers were **capped instead of removed**, as they represent real customers

---

## 🤖 Model Training & Evaluation
Multiple models were trained and compared, including:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- **CatBoost**
- AutoML using **PyCaret**

### Final Selected Model
After comparison and tuning, **CatBoost Classifier** performed best overall.

**Final Performance (approximate):**

| Metric | Value |
|------|------|
| ROC-AUC | ~0.85 |
| Recall | ~62% |
| Precision | ~61% |
| F1 Score | ~0.60 |

CatBoost was chosen because it:
- Handles categorical data efficiently
- Works well on imbalanced datasets
- Provides a good balance between recall and precision

---

## ⚙️ ML Pipeline
The project follows a clean and modular structure:

- Separate scripts for preprocessing and feature engineering
- Training and inference handled through a single pipeline (`main.py`)
- Trained model saved using `joblib`
- Threshold tuning applied based on business needs

---

## 🌐 Deployment & Threshold Control
The final model is deployed as a **Streamlit web application**.

### Deployment Highlights
- Real-time churn prediction
- Simple and clean user interface
- **Decision threshold slider** to control churn sensitivity

### Why Threshold Control?
- Lower threshold → catch more churners (higher recall)
- Higher threshold → fewer false alerts (higher precision)

This allows business teams to adjust predictions based on **cost, risk, and available resources**, which is how real production systems work.

---

## 🧰 Tools & Technologies
**Machine Learning & Data**
- Python, Pandas, NumPy
- Scikit-learn, PyCaret
- CatBoost, XGBoost, LightGBM
- Imbalanced-learn (SMOTE)

**Visualization**
- Matplotlib, Seaborn, Plotly

**Deployment**
- Streamlit
- Joblib

---

## 🚀 Key Learnings
- In churn problems, **recall is often more important than accuracy**
- Feature engineering can have a larger impact than changing models
- AutoML helps with quick comparison, but manual tuning is still necessary
- Choosing the right decision threshold is a **business decision**, not just a technical one

---

## 🔮 Future Improvements
- Try on diff-diff Datasets
- Add SHAP-based model explainability
- Implement cost-sensitive learning
- Deploy using FastAPI
- Monitor model performance over time
- Improve Batch Prediction

---

⭐ If you found this project useful, feel free to star the repository.

