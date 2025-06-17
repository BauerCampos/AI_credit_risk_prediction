# 📏 KNN - Credit Risk Prediction  

*A distance-based model achieving 90.1% accuracy by optimizing feature space and neighborhood size*

## 📌 Overview  
Implementation of a **K-Nearest Neighbors (KNN)** classifier to predict high-risk loans. Key wins:

    90.1% accuracy (best among tested configurations).

    Target encoding outperformed One-Hot by 6%, validating KNN’s sensitivity to the "curse of dimensionality".

    Precision-neighbor relationship: Accuracy plateaued beyond 10 neighbors (+0.1% at 15).

Business Impact: This model could help lenders reduce high-risk loan approvals, potentially saving millions in defaults.

## Dependencies
- pandas
- sklearn
- pickle
- yellowbrick

## ⚙️ Hyperparameters  
python
GaussianNB(
    n_neighbors=10,     # Optimal balance of precision and stability
    metric='minkowski,  # Euclidean distance (p=2) for isotropic feature space
    p = 2               # Confirmed Euclidean outperformed Manhattan (p=1)
)

**Why These values?**
    **n_neighbors=10**: Marginal gains (<0.1%) beyond 10 neighbors (see table).
    **Euclidean distance**: Assumed equal feature importance (scale-sensitive — scaling applied!).

## Preprocessing
**Data Cleaning**

    **Inconsistent Data**:Dropped illogical records(e.g. person_age / person_emp_length > 100)

    **Missing Data**:
    
        person_emp_length: Nulls were imputed with 0 (assuming nulls indicated no employment history).

        loan_int_rate: Nulls were replaced with the global mean, as these likely represented lost data.


**Encoding: Target vs One-Hot**

In this algorithm training the academical assumption of inverse proportionality between the number of dimensions and algorithm predictive power due to distortion of Euclidean distances was tested and proven as shown in the table below

|  Encoding  | Neighbors |  Accuracy  |  Insight                                        |
|------------|-----------|------------|-------------------------------------------------|
|  One-Hot   |    10     |   ~84.2%   |  High dimentinality hurt performance            |
|  Target    |    10     |   ~90.0%   |  Compact encoding improved neighbor similarity  |

**Neighbor Tuning**

|  n_neighbors  |  Accuracy(Target)  |         Trend         |
|---------------|--------------------|-----------------------|
|       3       |       ~88.6%       |  Underfitting         |
|      10       |       ~90.0%       |  Optimal trade-off    |
|      15       |       ~90.1%       |  Diminishing returns  |



## Key Takeaways

**Knn is computationally expensive**: larger number of neighbors costs more for minimal accuracy gains

**Dimensionality Matters**: Target encoding’s lower dimensionality was key for KNN success.

**Hyperparameter Sensitivity**: Small n_neighbors led to noise; large ones to over-smoothing.

## Performance Metrics

![Confusion Matrix](images/knn_cm.png)

![Classification Report](images/knn_cr.png)
