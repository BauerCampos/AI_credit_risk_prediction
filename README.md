# 🏦 Credit Risk Prediction Project

Predicting credit risk using machine learning algorithms. Developed as part of my AI specialization journey.

## 📌 Table of Contents
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Algorithms](#-Algorithms)
- [Results](#-results)
- [How to Run](#-how-to-run)
- [Future Improvements](#-future-improvements)
- [License](#-license)

## 🎯 Business Problem
**Objective:** Build a classifier to predict whether a bank customer is likely to default on a loan.

**Impact:**
- Reduce financial losses for banks
- Enable better loan approval decisions
- Minimize risk exposure

## 📊 Dataset
**Source:** [Credit Risk Dataset on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

**Features:**
- `person_age`: Customer age
- `person_income`: Annual Income
- `person_home_ownership`: Home ownership
- `person_emp_length`: Employment length (in years)
- `loan_intent`: Purpose of the loan
- `loan_grade`:Loan grade
- `loan_amnt`: Loan amount requested
- `loan_int_rate`: Interest rate
- `loan_percent_income`: Percent income
- `cb_person_default_on_file`: Historical default
- `cb_preson_cred_hist_length`: Credit history length
- `loan_status`: Target variable (0 = low risk, 1 = high risk)

## 🤖 Algorithms
- [Naïve Bayes](/aienv/algorithms/naive_bayes)
- [Decision Tree](/aienv/algorithms/decision_tree)
- [Random Forest](/aienv/algorithms/random_forest)
- [KNN](/aienv/algorithms/knn)
- [Logistic Regression](/aienv/algorithms/logistic_regression)
- [SVM](/aienv/algorithms/SVM)
- [Neural Network](/aienv/algorithms/neural_network)

## 📈 Results
**Performance Comparison:**

|         Model         | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Random Forest         |   0.935  |    0.94   |  0.94  |   0.93   |
| Neural Network        |   0.926  |    0.93   |  0.93  |   0.92   |
| SVM                   |   0.910  |    0.91   |  0.91  |   0.90   |
| KNN                   |   0.900  |    0.90   |  0.90  |   0.89   |
| Decision Tree         |   0.898  |    0.90   |  0.90  |   0.90   |
| Logistic Regression   |   0.868  |    0.86   |  0.87  |   0.86   |
| Naïve Bayes           |   0.845  |    0.85   |  0.85  |   0.85   |

## 🔧 Future Improvements
**Embeddings**
Embeddings are a powerful technique for encoding categorical variables, particularly effective for nominal data. Research and practical applications demonstrate their superiority over traditional methods (e.g., one-hot encoding) when preprocessing data for AI models, especially in deep learning.

**Multiple Imputation**
Two fields contained missing values: person_emp_length and loan_int_rate.

    person_emp_length: Nulls were imputed with 0 (assuming nulls indicated no employment history).

    loan_int_rate: Nulls were replaced with the global mean, as these likely represented lost data.

For higher accuracy, regression-based imputation (e.g., MICE) could predict missing values individually by leveraging relationships between variables.
