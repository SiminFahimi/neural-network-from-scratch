# Feedforward Neural Network from Scratch (NumPy)

## Overview

This project implements a fully connected feedforward neural network from scratch using only NumPy.

The goal is to solve a 3D nonlinear classification problem, where points are classified based on their position relative to a curved surface.

---

## Problem Definition

We generate random 3D data points (x, y, z) and classify them using the following decision boundary:

z > -(x - 3)^2 - (y - 5)^2 + 8

* Class 1 → points above the surface
* Class 0 → points below the surface

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