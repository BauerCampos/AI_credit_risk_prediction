# 🌳 Decision Tree - Credit Risk Prediction  

A robust, interpretable model achieving 89.8% accuracy in predicting loan defaults

## 📌 Overview  
Implementation of a Decision Tree classifier to assess credit risk, with data-driven optimizations. Key wins:

    89.8% accuracy (outperforming baseline models).

    One-Hot encoding surpassed Target encoding by 0.4% (unlike Naïve Bayes, where Target won).

    Minimal scaling impact (+0.03% accuracy gain), validating the algorithm’s resilience to feature scales.

Business Impact: This model could help lenders reduce high-risk loan approvals, potentially saving millions in defaults.

## Dependencies
- pandas
- sklearn
- pickle
- yellowbrick

## ⚙️ Hyperparameters  
python
DecisionTreeClassifier(
    criterion=entropy,           
    random_state = 0             # So results could be reproduced
)

## Preprocessing
**Data Cleaning**

    Inconsistent Data:Dropped illogical records(e.g. person_age / person_emp_length > 100)

    Missing Data:
    
        person_emp_length: Nulls were imputed with 0 (assuming nulls indicated no employment history).

        loan_int_rate: Nulls were replaced with the global mean, as these likely represented lost data.


**Encoding**

For this algorithm 2 methods were tested, Target encoding(with and without scailing) which had the lowest precision as expected since according to studies the algorithm is able to deal with high dimentionality (which is a potential down side of one hot), categorical variables and is less sensible to scailing given the fact that it analyzes every feature variable impact on the target variable independently

|     Scailing      |  Accuracy  |
|-------------------|------------|
|    No Scailing    |   ~89.39%  |
| Standard Scailing |   ~89.42%  |

Alternatively, one-hot encoding had the best performance as expected since the algorithm deals with the consequences caused by one hot very efficiently as showed below

|     Encoding      |  Accuracy  |
|-------------------|------------|
|  Target Encoding  |   ~89.4%   |
|      One-Hot      |   ~89.8%   |


## Key Takeaways

**Algorithm Fit**: Trees thrived with one-hot encoding unlike Naïve Bayes’ preference for Target.

**Real-World Readiness**: Minimal preprocessing (e.g., scaling) saves time in production.

**Interpretability**: Trees provide clear rules for lenders to justify rejections (great for compliance!).

## Performance Metrics

![Confusion Matrix](images/decision_tree_cm.png)

![Classification Report](images/decision_tree_cr.png)
