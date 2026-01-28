
The data is **highly imbalanced** (~80% non-churn, ~20% churn).

---

## ⚠️ 3. Why Accuracy is Misleading

If a model predicts:

> “Nobody will churn”

It already gets ~80% accuracy.

But it catches **zero churners** — which is useless.

So instead of accuracy, this project focuses on:

- **Recall** → How many churners did we detect?
- **AUC** → How well can we rank customers by churn risk?

---

## 🤖 4. AutoML Model Comparison (PyCaret)

We used **PyCaret AutoML** to compare multiple models.

| Model | Accuracy | AUC | Recall | F1 |
|------|--------|------|--------|----|
| Gradient Boosting | 0.82 | 0.74 | 0.30 | 0.40 |
| LightGBM | 0.82 | 0.73 | 0.32 | 0.41 |
| CatBoost | 0.82 | 0.73 | 0.31 | 0.41 |
| **Ridge Classifier** | 0.66 | 0.68 | **0.71** | **0.46** |

---

## 🔥 5. Key Insight

Although Gradient Boosting and CatBoost had higher accuracy, they only detected **~30% of churners**.

The Ridge model detected:

> **~71% of all churners**

That is more than **2× improvement in churn detection**.

---

## 🏦 6. Business Impact

Assume 1000 customers with 20% churn (200 customers).

| Model | Recall | Churners detected |
|------|--------|------------------|
| CatBoost | 0.31 | 62 |
| Ridge | **0.71** | **142** |

That means Ridge allows the bank to save **80 more customers**, which translates directly into revenue protection.

---

## 🧠 7. Why Ridge Has Lower AUC

Ridge is a **linear model**:
- It is very good at **detecting churners** (high recall)
- But weaker at **ranking customers by risk** (lower AUC)

This is acceptable because churn modeling is a **risk detection problem first, ranking problem second**.

---

## 🏗️ 8. Final Architecture (Industry-Style)

We use a **two-stage risk system**:

Stage 1 → Ridge Classifier
- High recall
- Catch most churners

Stage 2 → LightGBM
- High AUC
- Rank churners by risk


This is how real banks design churn systems.

---

## 🛠️ 9. Tech Stack

- Python  
- Pandas, NumPy  
- PyCaret (AutoML)  
- Scikit-learn  
- LightGBM  
- CatBoost  
- SHAP  

---

## 🏆 10. What This Project Demonstrates

This project shows understanding of:

- Class imbalance
- Business cost of false negatives
- Recall vs Accuracy trade-off
- Risk-based model selection
- Real-world churn modeling

This is **not a Kaggle accuracy project** — it is a **business-driven ML system**.

---

## 📈 11. Final Result

The final churn detection model achieves:

> **~71% recall**, allowing the bank to identify most customers who are about to leave and take proactive action.

---

⭐ If you find this useful, feel free to star the repository!
