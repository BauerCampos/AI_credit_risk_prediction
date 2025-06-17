# 🚀 SVM - Credit Risk Prediction  

*A high-performance model achieving 91.0% accuracy through optimal kernel selection and encoding strategies*

## 📌 Overview  
Implementation of a **Logistic Regression** classifier to predict high-risk loans. Key wins:

    91.0% accuracy (best among tested configurations).

    13.9% accuracy variance from kernel selection.

    Sensitiveness to high dimensions: Linear kernel training time with one-hot increased atleast 100 times compared to target encoding.

    Dual optimization: Scaled data reduced training time by 23% (13 sec → 10 sec)

Business Impact: This model could help lenders reduce high-risk loan approvals, potentially saving millions in defaults.

## Dependencies
- pandas
- sklearn
- pickle
- yellowbrick

## ⚙️ Hyperparameters  
python
SVC(
    kernel = 'rbf',    
    random_state = 1,
    c = 1.0
)

## Preprocessing
**Data Cleaning**

    Inconsistent Data:Dropped illogical records(e.g. person_age / person_emp_length > 100)

    Missing Data:
    
        person_emp_length: Nulls were imputed with 0 (assuming nulls indicated no employment history).

        loan_int_rate: Nulls were replaced with the global mean, as these likely represented lost data.


**Time Efficiency**

The algorithm had great results over a wide number of parameters and preprocessing configurations but the biggest impact was the model training time as consequence of the technique used to encode the categorical variables(One-Hot vs Target)

|  Encoding  |  Training Time  |  Accuracy  |                 Key Insight                    |
|------------|-----------------|------------|------------------------------------------------|
|  One-Hot   |     >15 min     |    N/A     |  Failed to Complete (curse of dimentionality)  |
|  Target    |     ~10 sec     |   91.0%    |  Avoided sparse feature explosion              |

Training halted after 15min for One-Hot - estimated **90-100x slower**

**Kernel Selection**

Kernel selection also played a big part on the algorithm efectiveness and training time with rbf kernel outperforming every other kernel in performance, time efficiency or both 

|   Kernel   |  Training Time  |  Accuracy  |
|------------|-----------------|------------|
|    RBF     |     ~10 sec     |   ~91.0%   |
|    Poly    |     ~10 sec     |   ~90.7%   |
|   Linear   |     ~12 sec     |   ~87.0%   |
|   Sigmoid  |     ~13 sec     |   ~77.1%   |


## Key Takeaways

**Encoding Dictates Feasibility**: Target encoding made training tractable vs One-Hot's dimensionality explosion.

**RBF Dominance**: 4% accuracy gain over linear kernel justified its use despite theoretical complexity.

**Scaling Benefits**:

    Accuracy: +9.5% (81.5% → 91.0%)

    Speed: 23% faster training (13 sec → 10 sec)

## Performance Metrics

![Confusion Matrix](images/svm_cm.png)

![Classification Report](images/svm_cr.png)
