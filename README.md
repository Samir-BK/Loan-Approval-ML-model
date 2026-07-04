# Loan Approval Prediction

A supervised machine learning project that predicts whether a loan application gets approved, based on applicant financial and demographic data.

## Dataset

`loan_approval_data.csv` — 1000 applicants, 19 columns (after dropping `Applicant_ID`).

Mix of numerical features (income, age, credit score, DTI ratio, savings, loan amount, etc.) and categorical features (employment status, marital status, loan purpose, property area, education, gender, employer category).

About 5% of rows had missing values across most columns.

Target class distribution: 62% not approved, 38% approved.

## Pipeline

1. **Missing values** — mean imputation for numerical columns, most-frequent imputation for categorical columns.
2. **EDA** — class balance, category counts, boxplots of key numerical features against approval status to spot outliers.
3. **Encoding** — label encoding for `Education_Level` and the target `Loan_Approved`; one-hot encoding (drop first) for the remaining categorical columns.
4. **Correlation check** — `Credit_Score` (+0.45) and `DTI_Ratio` (-0.44) stood out as the strongest predictors. Most other features barely correlate with approval.
5. **Split & scale** — 80/20 train-test split, `StandardScaler` fit on training data.
6. **Models** — Logistic Regression, KNN (k=7), and Gaussian Naive Bayes, each evaluated on precision, accuracy, recall, F1, and confusion matrix.
7. **Feature engineering** — added `DTI_Ratio_Sq` and `Credit_Score_Sq`, dropped the original `DTI_Ratio` and `Credit_Score` columns, then retrained all three models.

## Results

**Before feature engineering**

| Model | Precision | Accuracy | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.783 | 0.865 | 0.770 | 0.777 |
| KNN (k=7) | 0.609 | 0.745 | 0.459 | 0.523 |
| Naive Bayes | 0.804 | 0.865 | 0.738 | 0.769 |

**After feature engineering**

| Model | Precision | Accuracy | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.785 | 0.880 | 0.836 | 0.810 |
| KNN (k=7) | 0.646 | 0.765 | 0.508 | 0.569 |
| Naive Bayes | 0.811 | 0.860 | 0.705 | 0.754 |

Squaring `DTI_Ratio` and `Credit_Score` improved every model. Logistic Regression gained the most, picking up almost 6 points of F1.

## Conclusion

Naive Bayes has the best precision, so it's the safer choice if false approvals are costly. Logistic Regression has the best F1 and recall after feature engineering, so it catches more true approvals overall. KNN trails both across every metric.

## Requirements

```
pandas
numpy
seaborn
matplotlib
scikit-learn
```

## Usage

Run the notebook top to bottom. It expects `loan_approval_data.csv` in the same directory.
