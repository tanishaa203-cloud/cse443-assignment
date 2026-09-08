# Problem Set 02 — Bank Marketing: Term Deposit Subscription Prediction (Logistic Regression)

## Problem Statement
A banking institution wants to predict whether a customer will subscribe to a
term deposit based on their banking behaviour. The goal is to build a
**Logistic Regression** model using the "Bank Marketing Data Set", which
contains 17 attributes covering customer demographics and account details,
with a binary target: whether the customer subscribed (`yes`/`no`).

## Approach

1. **Data loading** — Loaded the dataset (semicolon-separated CSV, standard for
   the UCI Bank Marketing dataset).
2. **Exploration** — Checked for missing values, examined the class balance of
   the target variable (subscribed vs not subscribed), which is typically
   heavily skewed toward "no".
3. **Preprocessing:**
   - Encoded the target column (`yes` → 1, `no` → 0)
   - One-hot encoded categorical features (job, marital status, education,
     contact type, month, etc.)
   - Standardized numeric features with `StandardScaler`, since Logistic
     Regression is sensitive to feature scale
4. **Class imbalance handling** — Used `class_weight='balanced'` in the
   Logistic Regression model so the minority ("yes") class isn't ignored.
5. **Train/test split** — 80/20 split, stratified by the target to preserve
   class proportions in both sets.
6. **Model** — Logistic Regression (`scikit-learn`), trained on the scaled,
   encoded feature set.
7. **Evaluation** — Accuracy, precision, recall, F1-score, ROC-AUC, confusion
   matrix, and ROC curve on the held-out test set.
8. **Feature importance** — Examined the model's coefficients to identify which
   customer attributes most strongly influence the likelihood of subscribing.

## Repository Structure

```
problem_set_02/
├── bank_marketing_logistic_regression.ipynb   # Full notebook: load -> preprocess -> train -> evaluate
├── class_distribution.png                      # Target class balance (generated after running)
├── confusion_matrix.png                        # Test set confusion matrix (generated after running)
├── roc_curve.png                                # ROC curve (generated after running)
├── feature_importance.png                       # Top 15 influential features (generated after running)
└── README.md                                    # This file
```

## How to Run
1. Open `bank_marketing_logistic_regression.ipynb` in Google Colab.
2. Run all cells top to bottom — cell 1 downloads the dataset automatically.
3. Check cell 1's printed output; if the CSV path differs, adjust `DATA_PATH`
   in cell 3 (and the separator, `sep=';'` vs `sep=','`, if columns look wrong).

## Findings

**Test set results (9,043 customers):**

| Metric | Value |
|---|---|
| Accuracy | 0.8460 |
| Precision (yes) | 0.4186 |
| Recall (yes) | 0.8138 |
| F1-score (yes) | 0.5528 |
| ROC-AUC | 0.9079 |

**Classification report:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| no | 0.97 | 0.85 | 0.91 | 7,985 |
| yes | 0.42 | 0.81 | 0.55 | 1,058 |
| **Accuracy** | | | **0.85** | 9,043 |
| Macro avg | 0.70 | 0.83 | 0.73 | 9,043 |
| Weighted avg | 0.91 | 0.85 | 0.87 | 9,043 |

**Confusion Matrix:** see `confusion_matrix.png`
**ROC Curve:** see `roc_curve.png` (AUC = 0.908, indicating strong overall ranking ability)
**Top influential features:** see `feature_importance.png`

**Observations:**
- **ROC-AUC of 0.91** shows the model is very good at *ranking* customers by
  likelihood of subscribing, even though the dataset is heavily imbalanced
  (only ~11.7% of customers in the test set actually subscribed — 1,058 out of
  9,043).
- **Recall for "yes" is high (0.81)**: the model correctly identifies about 81%
  of customers who actually do subscribe. This is the direct effect of using
  `class_weight='balanced'` — without it, a model on this imbalanced data would
  likely just predict "no" for almost everyone and still get ~88% accuracy
  while catching almost none of the actual subscribers.
- **Precision for "yes" is low (0.42)**: of all the customers the model flags
  as likely to subscribe, only about 42% actually do. This is the trade-off
  that comes with prioritizing recall — the model casts a wide net.
- **Is this trade-off acceptable?** In a marketing context, this is often
  reasonable: the cost of contacting a customer who says "no" (a phone call)
  is much lower than the cost of missing a customer who would have said "yes"
  (a lost sale). A bank using this model to prioritize call lists would rather
  over-target than under-target, so high recall at the cost of some precision
  fits the business goal.
- **Accuracy (0.85) alone is a misleading metric here** because of the class
  imbalance — a naive model that always predicts "no" would score ~88%
  accuracy while being completely useless at identifying subscribers. This is
  exactly why precision, recall, and ROC-AUC matter more than raw accuracy for
  this problem.
- The top features driving predictions (see `feature_importance.png`) provide
  actionable insight for the bank's marketing team — e.g. call duration,
  whether a customer was contacted before, and the outcome of a previous
  campaign are typically the strongest predictors of subscription likelihood
  in this dataset.

## Limitations & Possible Improvements
- Logistic Regression assumes a linear relationship between features and
  log-odds of the outcome; a tree-based model (Random Forest, XGBoost) might
  capture non-linear patterns better, at the cost of interpretability.
- Feature selection or regularization tuning (L1/L2, adjusting `C`) could
  improve generalization.
- Cross-validation would give a more robust performance estimate than a single
  train/test split.
