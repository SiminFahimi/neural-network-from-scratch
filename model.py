import numpy as np
import copy


class FeedforwardNeuralNetwork:
    """
    Fully connected neural network implemented from scratch.

    Supports:
    - Multiple hidden layers
    - Sigmoid activation
    - Backpropagation
    - Gradient checking
    """

    def __init__(self, num_layers, num_features, hidden_units, num_classes,
                 activation, activation_prime):

        self.num_layers = num_layers
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_classes = num_classes

        self.activation = activation
        self.activation_prime = activation_prime

        self.theta = self._init_weights()

    # -------------------------
    # Initialization
    # -------------------------
    def _num_units(self, layer):
        if layer == 0:
            return self.num_features
        elif layer == self.num_layers - 1:
            return self.num_classes
        else:
            return self.hidden_units[layer]

    def _init_weights(self):
        """Initialize weights with small random values."""
        return [
            np.random.randn(self._num_units(l + 1), self._num_units(l) + 1) * 0.1
            for l in range(self.num_layers - 1)
        ]

    # -------------------------
    # Forward Pass
    # -------------------------
    def forward(self, X):
        activations = [X]
        z_values = [X]

        for l in range(1, self.num_layers):
            a_prev = activations[l - 1]
            a_prev = np.hstack([np.ones((a_prev.shape[0], 1)), a_prev])  # bias

            z = a_prev @ self.theta[l - 1].T
            a = self.activation(z)

            z_values.append(z)
            activations.append(a)

        return activations, z_values

    # -------------------------
    # Loss Function
    # -------------------------
    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        eps = 1e-12

        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m

        return loss

    # -------------------------
    # Backpropagation
    # -------------------------
    def backward(self, activations, z_values, y_true):
        m = y_true.shape[0]
        deltas = [None] * self.num_layers

        # Output layer error
        deltas[-1] = activations[-1] - y_true

        # Hidden layers
        for l in range(self.num_layers - 2, 0, -1):
            deltas[l] = (
                deltas[l + 1] @ self.theta[l][:, 1:]
            ) * self.activation_prime(z_values[l])

        grads = []
        for l in range(self.num_layers - 1):
            a = np.hstack([np.ones((m, 1)), activations[l]])
            grad = (deltas[l + 1].T @ a) / m
            grads.append(grad)

        return grads

    # -------------------------
    # Gradient Descent
    # -------------------------
    def update(self, grads, lr):
        for i in range(len(self.theta)):
            self.theta[i] -= lr * grads[i]

    # -------------------------
    # Training Loop
    # -------------------------
    def train(self, X, y, epochs=100, lr=5e-2):
        losses = []

        for epoch in range(epochs):
            activations, z = self.forward(X)
            loss = self.compute_loss(activations[-1], y)

            grads = self.backward(activations, z, y)
            self.update(grads, lr)

            losses.append(loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

        return losses

    # -------------------------
    # Prediction
    # -------------------------
    def predict(self, X):
        activations, _ = self.forward(X)
        return (activations[-1] > 0.5).astype(int)


# -------------------------
# Activation Functions
# -------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)
