from model import *
from plots import*
import numpy as np
from eval import test
from generate_data import data
from model import FeedforwardNeuralNetwork

def train(network, x_data, y_data, num_epochs, lr):
    final_cost_of_epochs = network.fit(x_data, y_data, num_epochs, lr)
    return final_cost_of_epochs


def compute_cost(network, x_data, y_data):
    activations, _ = network.forward_pass(x_data, network.weights)
    h = activations[-1]
    return network.binary_cross_entropy_loss(h, y_data, network.weights)



def cross_validation(x_data, y_data, fold_num, num_epochs, lr, net):
    x_val_folds = np.array_split(x_data, fold_num)
    y_val_folds = np.array_split(y_data, fold_num)

    fold_acc = []
    fold_cost = []

    for i in range(fold_num):
        net.weights = net.initialize_weights()

        x_val = x_val_folds[i]
        y_val = y_val_folds[i]

        x_train = np.concatenate([x_val_folds[j] for j in range(fold_num) if j != i])
        y_train = np.concatenate([y_val_folds[j] for j in range(fold_num) if j != i])

        net.fit(x_train, y_train, num_epochs, lr)

        acc, _ = test(net, x_val, y_val)
        cost = compute_cost(net, x_val, y_val)

        fold_acc.append(acc)
        fold_cost.append(cost)

    return np.mean(fold_acc), np.mean(fold_cost)



def lr_effection(dataset, learning_rates, fold_num, num_epochs, net):
    x_data, y_data, _, _, _, _ = dataset

    cv_accuracy = []
    cv_costs = []
    train_costs = []

    best_lr = None
    best_cost = float("inf")

    for lr in learning_rates:
        cv_acc, cv_cost = cross_validation(x_data, y_data, fold_num, num_epochs, lr, net)
        if best_cost > cv_cost:
            best_cost = cv_cost
            best_lr = lr
        cv_accuracy.append(cv_acc)
        cv_costs.append(cv_cost)

        net.weights = net.initialize_weights()
        net.fit(x_data, y_data, num_epochs, lr)
        final_cost = compute_cost(net, x_data, y_data)
        train_costs.append(final_cost)
        print(lr, "accurcy", cv_acc, "cost_cv", cv_cost, "cost_train", final_cost)

    print("\nBEST LR:", best_lr)

    curves = PlotCurve(
        data=[(learning_rates, cv_costs), (learning_rates, train_costs)],
        x_label="Learning Rate",
        y_label="Cost",
        line_label=["CV Cost", "Train Cost"],
        title="Learning Rate Comparison",
        save_address="results/lr_comparison.png",
    )
    plot_curves(curves)
    return best_lr



def cost_of_best_lr_each_epoch(dataset, best_lr, num_epochs, net):
    x_data, y_data, _, _, _, _ = dataset
    net.weights = net.initialize_weights()
    costs_of_each_epoch = net.fit(x_data, y_data, num_epochs, best_lr)

    epochs = range(1, num_epochs + 1)
    best_curve_plot = PlotCurve(
        data=[(epochs, costs_of_each_epoch)],
        x_label="Epoch",
        y_label="Cost",
        line_label=[f"lr={best_lr}"],
        title="Cost Curve on Best LR",
        save_address="results/cost_each_epochs.png",
    )
    plot_curves(best_curve_plot)

    return best_lr



def size_effection_on_cost(counts, fold_num, num_epochs, lr, net):
    cv_costs = []
    train_costs = []

    for n in counts:
        x_data, y_data, _, _, _, _ = data(n, data_eng= True)

        _cv_acc, cv_cost = cross_validation(x_data, y_data, fold_num, num_epochs, lr, net)
        cv_costs.append(cv_cost)

        net.weights = net.initialize_weights()
        net.fit(x_data, y_data, num_epochs, lr)
        train_costs.append(compute_cost(net, x_data, y_data))

    plot_curves(
        PlotCurve(
            data=[(counts, cv_costs), (counts, train_costs)],
            x_label="Number of Samples",
            y_label="Cost",
            line_label=["CV Cost", "Train Cost"],
            title="Effect of Data Size on Cost",
            save_address="results/effect_of_increasing_size.png",
        )
    )



def size_effection_on_accuracy(counts, num_epochs, lr, net):
    test_accs = []

    for n in counts:
        x_data, y_data, x_test, y_test, _, _ = data(n, data_eng= True)
net.weights = net.initialize_weights()
        net.fit(x_data, y_data, num_epochs, lr)

        acc, _ = test(net, x_test, y_test)
        test_accs.append(acc)

    plot_curves(
        PlotCurve(
            data=[(counts, test_accs)],
            x_label="Samples",
            y_label="Test Accuracy",
            line_label=["Accuracy"],
            title="Accuracy vs Data Size",
            save_address="results/accuracy_vs_size.png",
        )
    )