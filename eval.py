import numpy as np
from model import *

def test(network, x_test, y_test):
    val_predictions = network.predict_classes(x_test)
    acc = np.mean(val_predictions == y_test)
    return acc, val_predictions