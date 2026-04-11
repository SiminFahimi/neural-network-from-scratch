import numpy as np
import matplotlib.pyplot as plt
from generate_data import generate_train_data, generate_test_data
from model import FeedforwardNeuralNetwork, sigmoid, sigmoid_prime


def plot_3d_predictions(X, Y_true, Y_pred):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    correct = (Y_true == Y_pred).flatten()

    # True decision surface
    x = np.linspace(-10, 20, 50)
    y = np.linspace(-10, 20, 50)
    Xs, Ys = np.meshgrid(x, y)
    Zs = -(Xs - 3)**2 - (Ys - 5)**2 + 8

    ax.plot_surface(Xs, Ys, Zs, alpha=0.4)

    # Test points
    ax.scatter(
        X[correct, 0], X[correct, 1], X[correct, 2],
        s=5   
    )

    ax.scatter(
        X[~correct, 0], X[~correct, 1], X[~correct, 2],
        s=5   
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig("assets/prediction_3d.png")
    plt.title("Predictions + Decision Surface")
    plt.show()

# MAIN 

epoch = 300

X_train, Y_train = generate_train_data(10000)

NN = FeedforwardNeuralNetwork(
    num_layers=5,
    num_features=3,
    num_hidden_units={1:10, 2:10, 3:10},
    num_classes=1,
    activation_func=sigmoid,
    activation_func_prime=sigmoid_prime,
    theta=None
)

costs = NN.train(X_train, Y_train, epoch)
plt.plot(range(1, epoch + 1), costs, label="Cost")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.legend()
plt.grid(True)
plt.savefig("assets/costs.png")
plt.show()

print("Final Cost:", costs[-1])

X_test, Y_test = generate_test_data(2000)
output = NN.predict(X_test)

accuracy = (output == Y_test).mean()
print("Accuracy:", accuracy)


# Visualizations
plot_3d_predictions(X_test, Y_test, output)