import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    a = sigmoid(z)
    return a * (1 - a)


def relu(z):
    return np.maximum(0, z)


def relu_prime(z):
    return (z > 0).astype(float)