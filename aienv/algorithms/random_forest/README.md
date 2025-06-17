# 🌳 Random Forest - Credit Risk Prediction  

*A powerful ensemble model achieving 93.5% accuracy, outperforming simpler tree-based and probabilistic approaches*

## 📌 Overview  
Implementation of a Random Forest classifier to predict high-risk loans, leveraging ensemble learning for robust performance. Key achievements:

    93.5% accuracy (vs. 89.8% for Decision Tree and 84.5% for Naïve Bayes).
        
    Optimized n_estimators: Identified diminishing returns beyond 120 trees.

    Zero scaling/preprocessing overhead: Reused Decision Tree’s efficient pipeline.


Business Impact: A 3.7% accuracy boost over Decision Trees could reduce lender losses.

## Dependencies
- pandas
- sklearn
- pickle
- yellowbrick

## ⚙️ Hyperparameters  
python
RandomForestClassifier(
    criterion="entropy",     # Consistent with Decision Tree (for fair comparison)  
    n_estimators=120,       # Optimal trade-off: 93.52% accuracy (+0.12% vs. 100 trees)  
    random_state=0,         # Reproducibility  
)
  
## Preprocessing
**Data Cleaning**

    Inconsistent Data:Dropped illogical records(e.g. person_age / person_emp_length > 100)

    Missing Data:
    
        person_emp_length: Nulls were imputed with 0 (assuming nulls indicated no employment history).

        loan_int_rate: Nulls were replaced with the global mean, as these likely represented lost data.


## Preprocessing & Optimization

**Data Pipeline Reuse**

    Consistency: Applied identical cleaning/encoding as Decision Tree:

        One-Hot encoding (best performer in DT tests).

        No scaling (trees are scale-invariant).
       
    Efficiency: Saved hours by reusing proven preprocessing.

**Hyperparameter Tuning**

    |  N Estimators  |  Accuracy  |  Insight                     |
    |----------------|------------|------------------------------|
    |        80      |   ~93.40%  |  Baseline                    |
    |       100      |   ~93.50%  |  +0.1% accuracy              |
    |       120      |   ~93.52%  |  Diminishing returns (0.02%) |

## Key Takeaways

**Tradeoff awareness**: 120 trees balanced accuracy and computer cost

**Ensemble Advantage**: Random Forest’s bagging reduced overfitting vs. single Decision Tree.

**Efficiency Matters**: Reusing preprocessing accelerated experimentation.

## Performance Metrics

![Confusion Matrix](images/random_forest_cm.png)

![Classification Report](images/random_forest_cr.png)
