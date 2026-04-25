import numpy as np
import copy
from utils import *
class FeedforwardNeuralNetwork:
    def init(
        self,
        num_layers,
        num_features,
        num_hidden_units: list,
        num_classes,
        activation_func,
        activation_func_prime,
        weights=None,
    ):
        self.activation_func = activation_func
        self.activation_func_prime = activation_func_prime
        self.num_layers = num_layers

        self.num_features = num_features
        self.num_hidden_units = num_hidden_units
        self.num_classes = num_classes

        if weights is None:
            self.weights = self.initialize_weights()
        else:
            self.weights = weights

        self.output = None

    def he_initialization(self, l_idx):
        current_l_node_num = self.get_layer_size(l_idx)
        w = np.random.randn(self.get_layer_size(l_idx + 1), current_l_node_num + 1) * np.sqrt(
            2 / current_l_node_num
        )
        return w

    def xavier_init(self, l_idx):
        current_l_node_num = self.get_layer_size(l_idx)
        next_l_node_num = self.get_layer_size(l_idx + 1)

        limit = np.sqrt(6 / (current_l_node_num + next_l_node_num))
        w = np.random.uniform(-limit, limit, (next_l_node_num, current_l_node_num + 1))
        return w

    def initialize_weights(self):
        weights = []

        for i in range(self.num_layers - 1):
            if i == self.num_layers - 2:
                w = self.xavier_init(i)
            else:
                w = self.he_initialization(i)
            weights.append(w)

        return weights

    def get_layer_size(self, layer_idx):
        if layer_idx == 0:
            return self.num_features
        elif layer_idx == self.num_layers - 1:
            return self.num_classes
        elif 0 < layer_idx < self.num_layers - 1:
            return self.num_hidden_units[layer_idx - 1]
        else:
            raise IndexError

    def forward_pass(self, x_data, weights):
        z_list = []
        activations_list = []

        for layer_idx in range(self.num_layers):
            if layer_idx == 0:
                z_list.append(None)
                activations_list.append(x_data)
            else:
                pre_activation_no_bias = activations_list[layer_idx - 1]
                pre_activation = np.hstack(
                    [np.ones((pre_activation_no_bias.shape[0], 1)), pre_activation_no_bias]
                )

                curr_z = pre_activation.dot(weights[layer_idx - 1].T)
                z_list.append(curr_z)

                curr_activation = self.activation_func[layer_idx - 1](curr_z)
                activations_list.append(curr_activation)

        return activations_list, z_list

    def l2_regularization(self, lambda_, weights, m):
        regularization = 0
        for weight in weights:
            squared_weights = weight[:, 1:] ** 2
            regularization += np.sum(squared_weights)

        regularization = (lambda_ / (2 * m)) * regularization
        return regularization

    def binary_cross_entropy_loss(self, h_theta, y_data, weights, regularize=True, lambda_=1e-3):
        m = y_data.shape[0]

        eps = 1e-12
        h = np.clip(h_theta, eps, 1 - eps)

        cost = -np.sum(y_data * (np.log(h)) + (1 - y_data) * (np.log(1 - h))) / m

        if regularize:
            cost += self.l2_regularization(lambda_, weights, m)

        return cost

    def backpropagation(self, y_data, activations, z_list, weights):
        lambda_ = 1e-3
        g_prime = self.activation_func_prime
        num_layers = self.num_layers

        delta = [None] * num_layers
        delta[num_layers - 1] = activations[-1] - y_data

        for layer_idx in range(num_layers - 2, 0, -1):
            theta_l = weights[layer_idx]
            delta[layer_idx] = (delta[layer_idx + 1].dot(theta_l[:, 1:])) * g_prime[layer_idx - 1](
                z_list[layer_idx]
            )

        for _layer_idx in range(0, num_layers - 1):
            _ = np.zeros((self.get_layer_size(_layer_idx + 1), self.get_layer_size(_layer_idx) + 1))