import numpy as np

def generate_train_data(count: int):
    """
    Generate synthetic training data for a 3D classification problem.

    Decision boundary:
        z > -(x - 3)^2 - (y - 5)^2 + 8
    """
    x = np.random.uniform(-10, 20, size=(count, 1))
    y = np.random.uniform(-10, 20, size=(count, 1))
    z = np.random.uniform(-10, 20, size=(count, 1))

    X = np.hstack([x, y, z])
    y = (z > -(x - 3) ** 2 - (y - 5) ** 2 + 8).astype(int)

    return X, y


def generate_test_data(count: int):
    """Same as training data but used for evaluation."""
    return generate_train_data(count)
