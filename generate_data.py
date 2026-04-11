import numpy as np

def generate_train_data(count):
    x = np.random.uniform(-10, 20, size=(count, 1))
    y = np.random.uniform(-10, 20, size=(count, 1))
    z = np.random.uniform(-10, 20, size=(count, 1))

    X_train = np.hstack([x, y, z])
    y_train = (z > -(x - 3)**2 - (y - 5)**2 + 8).astype(int)

    return X_train, y_train


def generate_test_data(count):
    x = np.random.uniform(-10, 20, size=(count, 1))
    y = np.random.uniform(-10, 20, size=(count, 1))
    z = np.random.uniform(-10, 20, size=(count, 1))

    X_test = np.hstack([x, y, z])
    y_test = (z > -(x - 3)**2 - (y - 5)**2 + 8).astype(int)

    return X_test, y_test