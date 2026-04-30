from model import FeedforwardNeuralNetwork
from utils import *

def build_model(input_dim):
    return FeedforwardNeuralNetwork(
        num_layers=4,
        num_features=input_dim,
        num_hidden_units=[16, 8],
        num_classes=1,
        activation_func=[relu, relu, sigmoid],
        activation_func_prime=[relu_prime, relu_prime, sigmoid_prime],
        weights=None,
    )