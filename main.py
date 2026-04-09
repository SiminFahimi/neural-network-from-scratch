import matplotlib.pyplot as plt
from generate_data import generate_train_data, generate_test_data
from model import FeedforwardNeuralNetwork, sigmoid, sigmoid_prime


# Generate data
X_train, y_train = generate_train_data(10000)
X_test, y_test = generate_test_data(2000)

# Create model
model = FeedforwardNeuralNetwork(
    num_layers=5,
    num_features=3,
    hidden_units={1: 10, 2: 10, 3: 10},
    num_classes=1,
    activation=sigmoid,
    activation_prime=sigmoid_prime,
)

# Train model
losses = model.train(X_train, y_train, epochs=100)

# Evaluate
preds = model.predict(X_test)
accuracy = (preds == y_test).mean()
print(f"Accuracy: {accuracy:.4f}")

# Plot loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()
