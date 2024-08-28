import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize parameters
input_size = 1  # Single input feature
output_size = 1  # Single output neuron

# Learning rate
learning_rate = 0.1

# Initialize weights and bias
np.random.seed(0)
weight = np.random.randn(input_size, 1)  # Shape should be (1, 1)
bias = np.random.randn()

# Training data (simple binary classification)
X = np.array([[0], [1]])  # Input feature (shape: (2, 1))
y = np.array([[0], [1]])  # Output labels (shape: (2, 1))

# Training loop
epochs = 10000

for epoch in range(epochs):
    # Forward pass
    linear_output = np.dot(X, weight) + bias  # Shape: (2, 1) + Scalar
    predicted_output = sigmoid(linear_output)  # Shape: (2, 1)

    # Compute loss (Mean Squared Error)
    loss = np.mean((y - predicted_output) ** 2)

    # Backward pass
    output_error = y - predicted_output  # Shape: (2, 1)
    output_delta = output_error * sigmoid_derivative(predicted_output)  # Shape: (2, 1)

    # Update weights and biases
    weight_update = np.dot(X.T, output_delta).flatten()  # Shape: (1, 1) -> (1,)
    weight += weight_update * learning_rate  # Shape: (1,) + (1,) * Scalar

    bias += np.sum(output_delta) * learning_rate  # Scalar + Scalar

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}')

# Testing the network
print("\nPredictions:")
print(predicted_output)