# 🧠 Neural Network - Credit Risk Prediction  

*A high-performance MLP model achieving 92.5% accuracy through optimized architecture and preprocessing*

## 📌 Overview  
Implementation of a **Multi-Layer Perceptron (MLP)** for credit risk assessment. Key achievements:

    ~92.7% accuracy (best among tested configurations).

    Data Scaling ensured stable training vs not scaled erratic behavior (32-80% accuracy).

    Efficient Architecture: 2 hidden layers (16 neurons each) balanced speed and performance.


Business Impact: This model could help lenders reduce high-risk loan approvals, potentially saving millions in defaults.

## Dependencies
- pandas
- sklearn
- pickle
- yellowbrick

## ⚙️ Hyperparameters  
python
MLPClassifier(
    max_iter = 1500,            # Allowed convergence for complex patterns
    verbose = True,             # Monitored training progress
    tol = 0.000100,             # Default value
    solver = 'adam',            # Highest accuracy
    activation='relu',          # Avoided vanishing gradients (vs sigmoid/tanh)
    hidden_layer_sizes=(16,16)  # Optimized through testing(higher number of layers/neurons increased the model training time but not the consistent accuracy)
)

**Why this works**

    16×16 Layers: Deeper networks (e.g., 64×64) increased training time without consistent accuracy gains.

    Adam Optimizer: Handled sparse gradients from Target Encoding effectively.

## Preprocessing
**Data Cleaning**

    Inconsistent Data:Dropped illogical records(e.g. person_age / person_emp_length > 100)

    Missing Data:
    
        person_emp_length: Nulls were imputed with 0 (assuming nulls indicated no employment history).

        loan_int_rate: Nulls were replaced with the global mean, as these likely represented lost data.


**Scailing Stability**

Less than 10% of the not scaled database training went past the 35th (ranging from 14 to 50 iterations) iteration which resulted in a very sparse accuracy over the tests made (ranging from ~20% to ~70%), in contrast all of the scaled database training ranged from 200 to 280 iterations with the accuracy ranging from ~92.35% to ~92.55% with outliers going ~300 iterations and achieving ~92.70% accuracy

|  Scailing  |    Iterations(range)   |  Accuracy(range)  |                    Key Insight                       |
|------------|------------------------|-------------------|------------------------------------------------------|
|     No     |        14 - 72         |     32% - 80%     |  Failed to converge due to massive scale difference  |
|  Standard  |       200 - 280        |  92.35% - 92.55%  |  Reliable convergence; outliers hit 92.70%           |

Similar problem happened in logistic regression which emphasizes models similarity

**Encoding method difference**

One-hot performed marginally better with a average ~0.2% accuracy increase over Target encoding

|     Encoding      |  Accuracy  |
|-------------------|------------|
|      One-Hot      |   ~92.4%   |
|  Target Encoding  |   ~92.2%   |

## Key Takeaways

**Stability Signs**: Consistent iteration counts (200–280) indicated robust optimization.

**Small Can Be Powerful**: A 16×16 network outperformed larger models, proving efficiency matters.

## Performance Metrics

![Confusion Matrix](images/neural_network_cm.png)

![Classification Report](images/neural_network_cr.png)
