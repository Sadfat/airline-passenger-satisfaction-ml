# ✈️ Airline Passenger Satisfaction — Logistic Regression Analysis

Predict whether a passenger is **Satisfied** or **Dissatisfied** using a binomial Logistic Regression model, interpret coefficients to identify key satisfaction drivers, and deliver actionable airline recommendations.

---

## Dataset

| Property | Value |
|---|---|
| Rows | 129,880 |
| Features | 22 (service ratings, demographics, flight info) |
| Target | `satisfaction` → binary: `satisfied=1`, `dissatisfied=0` |
| Class balance | 54.7% satisfied · 45.3% dissatisfied |
| Missing values | 393 in `Arrival Delay in Minutes` — imputed with median |

---

## Model

**Binomial Logistic Regression** (`sklearn.linear_model.LogisticRegression`)  
Pipeline: `StandardScaler → LogisticRegression(solver='lbfgs', penalty='l2', C=1.0, max_iter=1000)`

---

## Model Assumptions Verified

| Assumption | Status |
|---|---|
| Binary outcome | ✅ Two classes: satisfied / dissatisfied |
| Independence of observations | ✅ One row per passenger, no repeated measures |
| No severe multicollinearity | ✅ No feature pairs with \|r\| > 0.75 |
| Adequate sample size | ✅ 103,904 training rows — 4,500× minimum requirement |

---

## Data Preparation & Encoding

| Feature | Encoding |
|---|---|
| `satisfaction` | `satisfied=1`, `dissatisfied=0` |
| `Customer Type` | Binary: `Loyal=1`, else `0` |
| `Type of Travel` | Binary: `Business=1`, else `0` |
| `Class` | One-hot: `Class_Business`, `Class_Eco`, `Class_Eco Plus` |
| Rating columns (0–5) | Used as-is (ordinal → continuous) |
| `Arrival Delay in Minutes` | Median imputation (right-skewed distribution) |

**Train/Test Split:** 80% / 20%, stratified, `random_state=42`

---

## Performance Metrics (Test Set: 25,976 passengers)

| Metric | Score |
|---|---|
| **Accuracy** | 82.9% |
| **Precision** | 84.4% |
| **Recall** | 84.3% |
| **F1** | 84.4% |
| **ROC-AUC** | 90.4% |

### Confusion Matrix

|  | Predicted: Dissatisfied | Predicted: Satisfied |
|---|---|---|
| **Actual: Dissatisfied** | 9,545 ✅ | 2,214 ❌ |
| **Actual: Satisfied** | 2,232 ❌ | 11,985 ✅ |

---

## Key Coefficients (Logistic Regression)

| Feature | Coefficient | Odds Ratio | Effect |
|---|---|---|---|
| Inflight Entertainment | +0.973 | 2.65 | Strongest positive driver |
| Customer Type (Loyal) | +0.725 | 2.06 | Loyal customers 2× more likely satisfied |
| On-board Service | +0.388 | 1.47 | Crew quality matters |
| Seat Comfort | +0.392 | 1.48 | Physical comfort equally important |
| Departure/Arrival Timing | −0.327 | 0.72 | Inconvenient scheduling hurts satisfaction |
| Food & Drink | −0.296 | 0.74 | Acts as active dissatisfier |

**Intercept:** Represents log-odds of satisfaction when all features are at their mean values.

---

## Business Recommendations

1. **Upgrade inflight entertainment** — largest single-feature impact (OR = 2.65)
2. **Fast-track loyalty programme** — loyal customers are 2× more likely to be satisfied
3. **Crew training + seat reconfiguration** on medium-haul economy routes
4. **Schedule audit** — timing perception directly reduces satisfaction odds by 28%
5. **Menu audit** — remove bottom-rated F&B items before adding premium options

---

## Repository Structure

```
├── logistic_regression_analysis.ipynb   # Full analysis — all cells executed
├── README.md                            # This file
└── airline_satisfaction.csv             # Source dataset
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook logistic_regression_analysis.ipynb
```
