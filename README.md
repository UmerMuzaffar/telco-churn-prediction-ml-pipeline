# 📉 Telco Customer Churn Prediction — Feature Selection × Oversampling × Classifier Pipeline

> Benchmarking 1,000 machine learning pipelines — 10 feature selection methods × 10 oversampling techniques × 10 classifiers — before and after hyperparameter tuning, to find the most accurate churn prediction model on an imbalanced telecom dataset.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat&logo=catboost&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-189A38?style=flat&logo=xgboost&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-Resampling-blue)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

📄 [Read the Full Project Report](report/Customer_Churn_Prediction_Report.pdf)

---

## The Problem

Customer churn — when a subscriber stops using a service — is one of the costliest problems in telecom. According to Bain & Company, increasing customer retention by just 5% can boost profits by 25–95%. Being able to flag at-risk customers early gives a business a direct, actionable lever for retention campaigns.

The Telco Customer Churn dataset reflects a challenge common to nearly every real-world churn problem: **severe class imbalance**. Only ~26.5% of customers churn, which biases naive models toward always predicting "no churn" — exactly the customers a business most needs to identify.

This project systematically tests **10 feature selection methods × 10 oversampling techniques × 10 classifiers** (1,000 combinations), evaluates each with stratified cross-validation, applies `GridSearchCV` tuning, and identifies the combination that delivers the best churn detection performance.

---

## Dataset

![Churn Distribution](images/churn_distribution.png)

| Attribute | Value |
|---|---|
| Source | Telco Customer Churn (IBM Sample Dataset, via Kaggle) |
| Samples (after cleaning) | 7,010 |
| Features | 19 predictors + target |
| Target | `Churn` — Yes / No |
| Non-churned (No) | 5,153 (73.5%) |
| Churned (Yes) | 1,857 (26.5%) |
| Class ratio | ≈ 74 : 26 |

**Preprocessing:**
- Dropped `customerID` (unique identifier, no predictive value)
- Converted `TotalCharges` from text/object to numeric, dropped resulting null rows
- Removed duplicate rows
- Binary categoricals (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `gender`) label-encoded to 0/1
- Multi-class categoricals (`Contract`, `InternetService`, `PaymentMethod`, etc.) one-hot encoded
- `StandardScaler` applied for most pipelines; `MinMaxScaler` applied specifically for Chi²-based selection (requires non-negative input)
- All scaling, selection, and resampling steps wrapped inside a single `imblearn` pipeline to prevent data leakage during cross-validation

---

## Exploratory Data Analysis

**Contract type is the strongest churn signal.** Month-to-month customers churn at **42.6%**, versus 11.3% for one-year contracts and just **2.8%** for two-year contracts — by far the clearest behavioral split in the dataset.

![Contract Type vs Churn](images/contract_vs_churn.png)

**Tenure tells the same story from a different angle.** Customers who stayed have a median tenure of **38 months**; those who churned had a median of just **10 months**.

![Tenure vs Churn](images/tenure_vs_churn.png)

**Payment method matters.** Electronic check users churn at **45.1%**, more than double the rate of automatic bank transfer (16.7%) or credit card (15.3%) users — convenience and automation correlate with retention.

![Payment Method vs Churn](images/payment_method_vs_churn.png)

**Family status correlates with retention.** Customers without a partner churn far more (1,188 vs 669), and customers without dependents churn far more (1,531 vs 326).

![Dependents vs Churn](images/dependents_vs_churn.png)

**Gender showed no meaningful difference** in churn rate (934 churned females vs 923 churned males), confirming it's a weak predictor on its own.

![Gender vs Churn](images/gender_vs_churn.png)

---

## Feature Selection Methods

10 filter-based univariate techniques were applied, each evaluating a feature's relationship to `Churn` independently:

| Method | Description |
|---|---|
| `SelectKBest (F-value)` | Top features by ANOVA F-test (linear dependency) |
| `SelectKBest (Chi²)` | Top features by chi-squared test (requires non-negative input, used with `MinMaxScaler`) |
| `SelectKBest (Mutual Info)` | Top features by mutual information (captures non-linear dependencies) |
| `SelectPercentile (F-value)` | Keeps the top 50th percentile of features by F-value |
| `SelectFpr (p-value)` | Controls false positive rate at α = 0.05 |
| `SelectFwe (Family-wise Error)` | Controls family-wise error rate at α = 0.05 |
| `SelectFdr (False Discovery Rate)` | Controls false discovery rate at α = 0.05 |
| `Variance Threshold` | Removes near-constant features (threshold = 0.01) |
| `GenericUnivariateSelect (F)` | F-value scoring, percentile mode |
| `GenericUnivariateSelect (Mutual Info)` | Mutual information scoring, k-best mode |

---

## Oversampling Techniques

10 methods were used to address the 74:26 class imbalance, all applied **inside the training folds only** to avoid leakage:

**From `imbalanced-learn`:**

| Method | Description |
|---|---|
| `ADASYN` | Adaptive synthetic sampling — generates more samples in hard-to-learn regions |
| `BorderlineSMOTE` | SMOTE variant focused on borderline examples between classes |
| `KMeansSMOTE` | Clusters minority instances, applies SMOTE within each cluster |
| `SMOTETomek` | Hybrid of SMOTE oversampling and Tomek Link undersampling |

**From `smote-variants`:**

| Method | Description |
|---|---|
| `DistanceSMOTE` | Distance-weighted synthetic point generation |
| `V_SYNTH` | Virtual sample synthesis for improved generalization |
| `ROSE` | Random Over-Sampling Examples — smoothed bootstrap |
| `PDFOS` | Probability Density Function Oversampling (kernel density estimation) |
| `MDO` | Majority-Directed Oversampling — generates points influenced by majority class geometry |
| `polynom_fit_SMOTE_star` | Fits polynomial surfaces to generate structurally meaningful synthetic points |

---

## Classifiers

10 classifiers were trained and evaluated using **Stratified 5-Fold Cross-Validation**:

Logistic Regression · Decision Tree · Random Forest · Gradient Boosting · AdaBoost · CatBoost · Naive Bayes · K-Nearest Neighbors · XGBoost · Linear Discriminant Analysis (LDA)

---

## Results — Before Hyperparameter Tuning

All 1,000 combinations (10 selectors × 10 oversamplers × 10 classifiers) were evaluated with default hyperparameters using Stratified 5-Fold Cross-Validation.

### Top 10 Combinations (Baseline)

| Rank | Selector | Oversampler | Classifier | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1 | Variance Threshold | ROSE | GradientBoosting | 0.8068 | 0.6611 | 0.5557 | 0.6037 |
| 2 | SelectFpr (p-value) | ROSE | GradientBoosting | 0.8061 | 0.6582 | 0.5584 | 0.6040 |
| 3 | SelectFpr (p-value) | polynom_fit_SMOTE_star | GradientBoosting | 0.8061 | 0.6652 | 0.5396 | 0.5957 |
| 4 | SelectFdr (FDR) | ROSE | GradientBoosting | 0.8060 | 0.6600 | 0.5514 | 0.6006 |
| 5 | SelectFdr (FDR) | polynom_fit_SMOTE_star | GradientBoosting | 0.8060 | 0.6648 | 0.5396 | 0.5955 |
| 6 | SelectFpr (p-value) | PDFOS | GradientBoosting | 0.8057 | 0.6590 | 0.5525 | 0.6009 |
| 7 | SelectFwe (FWE) | ROSE | GradientBoosting | 0.8056 | 0.6588 | 0.5514 | 0.6002 |
| 8 | SelectFpr (p-value) | V_SYNTH | GradientBoosting | 0.8056 | 0.6649 | 0.5363 | 0.5935 |
| 9 | SelectFdr (FDR) | MDO | GradientBoosting | 0.8050 | 0.6623 | 0.5380 | 0.5936 |
| 10 | Variance Threshold | MDO | GradientBoosting | 0.8047 | 0.6603 | 0.5412 | 0.5948 |

### Best Before Tuning

```
🏆 Best by Accuracy:
Selector       Variance Threshold
Oversampler    ROSE
Classifier     GradientBoosting
Accuracy       0.8068
Precision      0.6611
Recall         0.5557
F1             0.6037

🏆 Best by Precision:
Selector       SelectKBest (F-value)
Oversampler    ROSE
Classifier     AdaBoost
Precision      0.6720

🏆 Best by Recall:
Selector       SelectKBest (Chi²)
Oversampler    KMeansSMOTE
Classifier     NaiveBayes
Recall         0.9316

🏆 Best by F1 Score:
Selector       SelectFdr (False Discovery Rate)
Oversampler    SMOTETomek
Classifier     GradientBoosting
F1             0.6365
```

**GradientBoosting + Variance Threshold + ROSE** led the baseline pass at 0.8068 accuracy — but the top ~10 configurations were tightly clustered (0.804–0.807), all dominated by GradientBoosting paired with statistical filter selectors and domain-aware oversamplers.

---

## Hyperparameter Tuning

`GridSearchCV` (wrapped inside the full pipeline to avoid leakage) was applied to the strongest combinations, using **Stratified 5-Fold Cross-Validation** for every candidate parameter set.

| Classifier | Hyperparameters Tuned |
|---|---|
| Logistic Regression | `C`: [0.01, 0.1, 1, 10] · `penalty`: l2 · `solver`: liblinear |
| Decision Tree | `max_depth`: [5, 10, None] · `min_samples_split`: [2, 5] · `min_samples_leaf`: [1, 2] |
| Random Forest | `n_estimators`: [100, 200] · `max_depth`: [10, 20, None] · `min_samples_split`: [2, 5] · `min_samples_leaf`: [1, 2] |
| Gradient Boosting | `n_estimators`: [100, 200] · `learning_rate`: [0.01, 0.1] · `max_depth`: [3, 5] |
| AdaBoost | `n_estimators`: [50, 100] · `learning_rate`: [0.1, 0.5, 1.0] · base estimator: DecisionTree (depth 1) |
| CatBoost | `iterations`: [100, 200] · `learning_rate`: [0.05, 0.1] · `depth`: [4, 6] · `l2_leaf_reg`: [1, 3] · `border_count`: [32, 64] |
| Naive Bayes | No tunable hyperparameters (default `GaussianNB`) |
| KNN | `n_neighbors`: [3, 5, 7, 9] · `weights`: [uniform, distance] |
| XGBoost | `n_estimators`: [50, 100] · `learning_rate`: [0.05, 0.1] · `max_depth`: [3, 5] |
| LDA | `solver`: [svd, lsqr] · `shrinkage`: [None, auto] |

### Accuracy Shift After Tuning

| Combination | Before Tuning | After Tuning | Change |
|---|---|---|---|
| SelectFpr (p-value) + MDO | — | **0.8091** (CatBoost) | New top result |
| Variance Threshold + ROSE | 0.8068 (GradientBoosting) | 0.8068 (GradientBoosting) | 0.0000 |
| SelectFdr (FDR) + SMOTETomek | 0.6365 F1 (GradientBoosting) | 0.6358 F1 (AdaBoost) | Best-F1 model changed |

> Tuning shifted the overall winner from **GradientBoosting** (baseline best, 0.8068) to **CatBoost** (tuned best, 0.8091) — a +0.23 percentage point gain driven by CatBoost's depth/border-count tuning combined with `SelectFpr` and `MDO`.

---

## Results — After Hyperparameter Tuning

### Top 15 Combinations (Tuned)

| Rank | Selector | Oversampler | Classifier | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1 | SelectFpr (p-value) | MDO | **CatBoost** | **0.8091** | 0.6773 | 0.5337 | 0.5968 |
| 2 | SelectFpr (p-value) | V_SYNTH | CatBoost | 0.8084 | 0.6792 | 0.5240 | 0.5914 |
| 3 | Variance Threshold | polynom_fit_SMOTE_star | CatBoost | 0.8077 | 0.6761 | 0.5256 | 0.5914 |
| 4 | SelectFwe (FWE) | ROSE | GradientBoosting | 0.8071 | 0.6670 | 0.5428 | 0.5980 |
| 5 | Variance Threshold | ROSE | CatBoost | 0.8068 | 0.6742 | 0.5245 | 0.5899 |
| 6 | SelectFdr (FDR) | V_SYNTH | CatBoost | 0.8068 | 0.6762 | 0.5196 | 0.5876 |
| 7 | SelectFdr (FDR) | ROSE | CatBoost | 0.8068 | 0.6738 | 0.5245 | 0.5896 |
| 8 | SelectFpr (p-value) | polynom_fit_SMOTE_star | CatBoost | 0.8067 | 0.6742 | 0.5229 | 0.5888 |
| 9 | SelectFdr (FDR) | polynom_fit_SMOTE_star | CatBoost | 0.8067 | 0.6742 | 0.5229 | 0.5888 |
| 10 | SelectFdr (FDR) | MDO | CatBoost | 0.8067 | 0.6716 | 0.5283 | 0.5911 |
| 11 | Variance Threshold | V_SYNTH | CatBoost | 0.8064 | 0.6746 | 0.5202 | 0.5873 |
| 12 | SelectFpr (p-value) | polynom_fit_SMOTE_star | GradientBoosting | 0.8061 | 0.6652 | 0.5396 | 0.5957 |
| 13 | SelectFwe (FWE) | ROSE | CatBoost | 0.8060 | 0.6716 | 0.5229 | 0.5877 |
| 14 | SelectFpr (p-value) | PDFOS | CatBoost | 0.8060 | 0.6724 | 0.5218 | 0.5873 |
| 15 | SelectFdr (FDR) | polynom_fit_SMOTE_star | GradientBoosting | 0.8060 | 0.6648 | 0.5396 | 0.5955 |

### Best & Worst After Tuning

```
🏆 Best by Accuracy:
Selector       SelectFpr (p-value)
Oversampler    MDO
Classifier     CatBoost
Accuracy       0.8091
Precision      0.6773
Recall         0.5337
F1             0.5968
BestParams     {'border_count': 64, 'depth': 4, 'iterations': 200, ...}

🏆 Best by Precision:
Selector       SelectFdr (False Discovery Rate)
Oversampler    MDO
Classifier     RandomForest
Precision      0.6844

🏆 Best by Recall:
Selector       SelectKBest (F-value)
Oversampler    polynom_fit_SMOTE_star
Classifier     NaiveBayes
Recall         0.9182

🏆 Best by F1 Score:
Selector       SelectFdr (False Discovery Rate)
Oversampler    SMOTETomek
Classifier     AdaBoost
F1             0.6358

⚠️ Worst by Accuracy:
Selector       SelectFpr (p-value)
Oversampler    V_SYNTH
Classifier     LogisticRegression
Accuracy       0.5275
Precision      0.2681
Recall         0.5059
F1             0.3477
```

### Confusion Matrix — Best Pipeline (SelectFpr + MDO + CatBoost)

![Confusion Matrix - Best Pipeline](images/best_pipeline_confusion_matrix.png)

| | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | 926 (TN) | 104 (FP) |
| **Actual Yes** | 160 (FN) | 212 (TP) |

The model correctly identified 212 of 372 actual churners (57% recall) while keeping false positives low (104 out of 1,030 non-churners) — a balanced trade-off for a 74:26 imbalanced dataset.

---

## Aggregate Performance Views

### Average Accuracy per Classifier (across all selectors & oversamplers, tuned)

![Average Accuracy per Classifier](images/avg_accuracy_per_classifier.png)

| Classifier | Avg. Accuracy |
|---|---|
| GradientBoosting | 0.7759 |
| **CatBoost** | 0.7737 |
| RandomForest | 0.7724 |
| XGBoost | 0.7724 |
| AdaBoost | 0.7643 |
| DecisionTree | 0.7546 |
| KNN | 0.7439 |
| LDA | 0.7210 |
| LogisticRegression | 0.7149 |
| NaiveBayes | 0.6370 |

### Best Classifier per Feature Selector

![Best Classifier per Feature Selector](images/best_classifier_per_selector.png)

| Feature Selector | Best Classifier | Accuracy |
|---|---|---|
| SelectFpr (p-value) | CatBoost | **0.8091** |
| Variance Threshold | CatBoost | 0.8077 |
| SelectFwe (FWE) | GradientBoosting | 0.8071 |
| SelectFdr (FDR) | CatBoost | 0.8068 |
| GenericUnivariateSelect (F) | CatBoost | 0.8026 |
| SelectPercentile (F-value) | CatBoost | 0.8026 |
| SelectKBest (Mutual Info) | CatBoost | 0.7981 |
| GenericUnivariateSelect (Mutual Info) | CatBoost | 0.7973 |
| SelectKBest (F-value) | AdaBoost | 0.7949 |
| SelectKBest (Chi²) | DecisionTree | 0.7596 |

### Best Classifier per Oversampler

![Best Classifier per Oversampler](images/best_classifier_per_oversampler.png)

| Oversampler | Best Classifier | Accuracy |
|---|---|---|
| MDO | CatBoost | **0.8091** |
| V_SYNTH | CatBoost | 0.8084 |
| polynom_fit_SMOTE_star | CatBoost | 0.8077 |
| ROSE | GradientBoosting | 0.8071 |
| PDFOS | CatBoost | 0.8060 |
| DistanceSMOTE | GradientBoosting | 0.8003 |
| KMeansSMOTE | CatBoost | 0.7942 |
| ADASYN | CatBoost | 0.7932 |
| SMOTETomek | CatBoost | 0.7894 |
| BorderlineSMOTE | CatBoost | 0.7887 |

### Accuracy Heatmap — Oversampler × Classifier

![Accuracy Heatmap](images/accuracy_heatmap_oversampler_classifier.png)

Naive Bayes (rightmost dark column) consistently underperformed regardless of oversampler — the only classifier never breaking 0.66 accuracy. MDO, ROSE, PDFOS, V_SYNTH, and polynom_fit_SMOTE_star produced the strongest scores across nearly every other classifier, while ADASYN and BorderlineSMOTE consistently lagged.

---

## Key Findings

**CatBoost + SelectFpr + MDO is the best overall combination** — 0.8091 accuracy, 0.5968 F1, edging out the next-best configuration by a small but consistent margin.

**Tuning changed the winner.** Before tuning, GradientBoosting + Variance Threshold + ROSE led at 0.8068. After `GridSearchCV` tuning, CatBoost overtook it at 0.8091 — a reminder that the "best" untuned model isn't always the best after optimization.

**CatBoost dominated post-tuning** — it was the top classifier for 7 of 10 feature selectors and 7 of 10 oversamplers, the most consistent performer across the entire grid.

**Statistical filter-based selectors outperformed SelectKBest variants.** `SelectFpr`, `SelectFdr`, `SelectFwe`, and `Variance Threshold` all clustered around 0.806–0.809 accuracy, while `SelectKBest (Chi²)` was the clear laggard (best accuracy only 0.7596).

**Domain-aware synthetic oversamplers beat classical SMOTE variants.** `MDO`, `V_SYNTH`, `polynom_fit_SMOTE_star`, `ROSE`, and `PDFOS` consistently outperformed `ADASYN`, `BorderlineSMOTE`, and `SMOTETomek` across most classifiers.

**Naive Bayes was the clear outlier** — average accuracy of 0.6370, the only classifier never competitive with the ensemble methods, though it achieved the **highest recall of any model (0.9182–0.9316)** when paired with `SelectKBest (Chi²)` or `SelectKBest (F-value)` — useful only if the business goal is "never miss a churner" regardless of false alarms.

**Logistic Regression + V_SYNTH was the single worst combination** (0.5275 accuracy) — the linear decision boundary couldn't handle the synthetic samples generated by V_SYNTH's virtual sample synthesis approach.

---

## Final Model

| | |
|---|---|
| **Feature Selector** | SelectFpr (p-value, α = 0.05) |
| **Oversampler** | MDO (Majority-Directed Oversampling) |
| **Classifier** | CatBoost (tuned) |
| **Accuracy** | 0.8091 |
| **Precision** | 0.6773 |
| **Recall** | 0.5337 |
| **F1 Score** | 0.5968 |
| **Best Params** | `border_count=64, depth=4, iterations=200, learning_rate=0.05, l2_leaf_reg=1` |

This combination is recommended as the foundation for a production churn-scoring pipeline. It balances detection of at-risk customers (53% recall) with low false-alarm rates (68% precision), suitable for prioritizing retention outreach without overwhelming customer success teams with false positives.

---

## Repository Structure

```
telco-churn-prediction-ml-pipeline/
│
├── notebooks/
│   └── customer_churn_fs_os_pipeline.ipynb
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── images/
│   ├── churn_distribution.png
│   ├── contract_vs_churn.png
│   ├── tenure_vs_churn.png
│   ├── payment_method_vs_churn.png
│   ├── gender_vs_churn.png
│   ├── dependents_vs_churn.png
│   ├── best_pipeline_confusion_matrix.png
│   ├── avg_accuracy_per_classifier.png
│   ├── best_classifier_per_selector.png
│   ├── best_classifier_per_oversampler.png
│   └── accuracy_heatmap_oversampler_classifier.png
│
├── report/
│   └── Customer_Churn_Prediction_Report.pdf
│
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

---

## Setup & Usage

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
smote-variants
catboost
xgboost
tqdm
```

Open `notebooks/customer_churn_fs_os_pipeline.ipynb` and run cells in order:

1. **Cells 1–10** — Load data, drop `customerID`, fix `TotalCharges` dtype, drop nulls/duplicates
2. **Cells 11–17** — Exploratory data analysis: gender, partner, dependents, phone service, tenure, monthly/total charges, contract type, payment method vs churn
3. **Cells 18–23** — Encode categorical variables (label encoding + one-hot), separate features/target
4. **Cells 24–25** — Class imbalance visualization (bar chart + pie chart)
5. **Cells 26–28** — Define the 10 feature selectors, 10 oversamplers, and 10 classifiers
6. **Cells 29–32** — Run all 1,000 baseline pipeline combinations with Stratified 5-Fold CV, display results sorted by accuracy
7. **Cell 33** — Define `GridSearchCV` hyperparameter grids for all 10 classifiers
8. **Cells 34–37** — Run hyperparameter-tuned pipelines, display best/worst results
9. **Cells 38–47** — Generate confusion matrix, accuracy/F1 comparison charts, and heatmaps

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| ML models | scikit-learn, CatBoost, XGBoost |
| Feature selection | scikit-learn (`SelectKBest`, `SelectFpr`, `SelectFdr`, `SelectFwe`, `SelectPercentile`, `GenericUnivariateSelect`, `VarianceThreshold`) |
| Oversampling | imbalanced-learn (`ADASYN`, `BorderlineSMOTE`, `KMeansSMOTE`, `SMOTETomek`), smote-variants (`DistanceSMOTE`, `V_SYNTH`, `ROSE`, `PDFOS`, `MDO`, `polynom_fit_SMOTE_star`) |
| Hyperparameter tuning | GridSearchCV |
| Validation | Stratified 5-Fold Cross-Validation |
| Visualization | Matplotlib, Seaborn |

---

## About

Machine learning churn prediction — Telco Customer Churn dataset (7,010 samples)
10 feature selection methods × 10 oversampling techniques × 10 classifiers × before/after tuning comparison (1,000 pipelines)
Best result: 80.91% accuracy, 59.68% F1 (CatBoost + SelectFpr + MDO)
