from plots import *
from generate_data import data
from eval import *
from train import *

def main():
    counts = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    fold_num = 4

    dataset = data(2000, data_eng = True)
    x_data, y_data, x_test, y_test, x_mean, x_std = dataset

    learning_rates = [
        0.001,
        0.002,
        0.004,
        0.008,
        0.016,
        0.032,
        0.064,
        0.128,
        0.256,
        0.512,
        1.024,
        2.048,
        4.096,
        9.192,
    ]
    num_epochs = 100

    net = FeedforwardNeuralNetwork(
        num_layers=4,
        num_features=x_data.shape[1],
        num_hidden_units=[32, 16],
        num_classes=1,
        activation_func=[relu, relu, sigmoid],
        activation_func_prime=[relu_prime, relu_prime, sigmoid_prime],
        weights=None,
    )

    best_lr = lr_effection(dataset, learning_rates, fold_num, num_epochs, net)
    size_effection_on_cost(counts, fold_num, num_epochs, best_lr, net)
    size_effection_on_accuracy(counts, num_epochs, best_lr, net)
    cost_of_best_lr_each_epoch(dataset,best_lr, num_epochs,net)

    net.weights = net.initialize_weights()
    net.fit(x_data, y_data, num_epochs, best_lr)
    test_acc, test_prediction = test(net, x_test, y_test)
    print("TEST ACC:", test_acc)
    print(np.sum(y_data == 0), np.sum(y_data == 1))

    plot_3d_predictions(x_test, y_test, test_prediction, x_std, x_mean)

if __name__ == "__main__":
    main()