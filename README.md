# Feedforward Neural Network from Scratch (NumPy)

## Overview

This project was implemented as a self-learning exercise to gain a deeper understanding of neural networks.

It focuses on building a feedforward neural network from scratch using NumPy, including forward propagation, backpropagation, and gradient descent, without relying on deep learning frameworks.

---

## Problem Definition

We generate random 3D data points (x, y, z) and classify them based on their position relative to a nonlinear surface:

z > -(x - 3)^2 - (y - 5)^2 + 8

The task is essentially to determine whether each point lies **above or below a curved surface (paraboloid)**.

* Class 1 → points above the surface
* Class 0 → points below the surface

This can be interpreted as a geometric problem of separating points **above and under a 3D curve**, making it a nonlinear classification task.

---

## Features

* Fully implemented neural network (no ML frameworks)
* Forward propagation
* Backpropagation
* Gradient descent optimization
* Gradient checking (numerical verification)
* L2 regularization
* Custom data generation
* Training visualization (loss curve)

---

## Model Architecture

* Input layer: 3 features (x, y, z)
* Hidden layers: 3 layers (10 neurons each)
* Output layer: 1 neuron (binary classification)

Activation function: Sigmoid

---

## Training

* Loss function: Binary Cross-Entropy
* Learning rate: 0.01
* Epochs: 100+
* Optimization: Batch Gradient Descent

---

## Results

* The model successfully learns a nonlinear decision boundary
* Loss decreases over training iterations
* Achieves good accuracy on test data (~85–95%)

---

## How to Run

```bash
python main.py
```

---

## Dependencies

* numpy
* matplotlib

---

## Future Improvements

* Add ReLU activation (for faster training)
* Implement mini-batch gradient descent
* Add Adam optimizer
* Improve weight initialization (Xavier/He)
* Visualize decision boundary in 3D

## Training Loss

![Loss Curve](images/plot_loss.png)
![accuracy](images/loss.png)
