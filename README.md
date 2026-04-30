# Neural Network from Scratch (NumPy)

A feedforward neural network implemented entirely with **Python + NumPy**, without using frameworks such as PyTorch or TensorFlow.

This personal learning project was built to understand the internal mechanics of neural networks by implementing every major component manually, including forward propagation, backpropagation, optimization, and evaluation.

---

## What This Project Covers

- Fully connected feedforward neural network  
- Forward pass implementation from scratch  
- Backpropagation from scratch  
- Binary cross-entropy loss  
- Gradient descent optimization  
- L2 regularization  
- Cross-validation for model selection  
- Experiment pipeline for testing:
  - Learning rate  
  - Regularization strength  
  - Dataset size  

---

## Dataset

A synthetic binary classification dataset was generated using two input variables:

```python
x ~ Uniform(-10, 10)
y ~ Uniform(-10, 10)
```

A nonlinear decision boundary was defined as:

```python
f(x, y) = sin(x) + cos(y) + 0.15 * (x^2 + y^2) + 0.5 * sin(xy / 4)
```

Labels were assigned based on whether samples were above or below this surface, with added noise.

---

## Feature Engineering

Based on the structure of the dataset, several engineered features were added:

- \( x^2 \)  
- \( y^2 \)  
- \( \sin(x) \)  
- \( \cos(x) \)  

---

## Results

### Raw vs Engineered Features

| Input Type          | Accuracy |
|--------------------|----------|
| Raw Features       | 0.9587   |
| Engineered Features| 0.9663   |

The performance gain was small but consistent.

## Training Curve
![cost each epoch](results/loss.png)

## Effect of size
![accuracy vs size](results/size_effect_on_accuracy.png)

![effect of increasing size](results/size_effect_on_cost.png)

## Lamba comparison
![Lamba comparison](results/lambda_effect.png)

## Learning rate comparison
![lr comparison](results/lr_effect.png)

## 3D Predictions
![Predictions](results/prediction_3d.png)

---

## Experiments Conducted

- Hyperparameter tuning  
- Learning rate comparison  
- L2 regularization analysis  
- Dataset size impact  
- Learning curves  
- 3D decision boundary visualization  

---

## Project Structure
```
.
├── model.py # Neural network implementation
├── train.py # Training pipeline
├── evaluate.py # Model evaluation
├── plots.py # Visualization utilities
├── data.py # Synthetic dataset generation
├── results/
│ ├── loss.png
│ ├── lr_effect.png
│ ├── lambda_effect.png
│ ├── size_effect_on_accuracy.png
│ ├── size_effect_on_cost.png
│ ├── prediction_3d.png
└── README.md
```
---

## Implementation Notes

- Implemented using NumPy only
- No external machine learning libraries used
- Focused on correctness, clarity, and learning fundamentals

---

## Possible Extensions

- Adam optimizer
- RMSProp optimizer
- Mini-batch gradient descent
- Deeper neural networks
- Multi-class classification
- PyTorch reimplementation for benchmarking
- Automated feature selection

---

## Summary

This project explores:

- Neural networks from first principles
- The effect of feature representation
- The impact of hyperparameters on training behavior