# ---------------------------------------------
# Hyper-Parameter Tuning for BPNN (XOR Problem)
# ---------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# 1. XOR Dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# 2. Hyperparameters
learning_rate = 0.1
epochs = 5000
hidden_neurons = 4

# 3. Initialize Weights
np.random.seed(42)

W1 = np.random.randn(2, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(hidden_neurons, 1)
b2 = np.zeros((1, 1))

# 4. Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 5. Training using Backpropagation
losses = []

for epoch in range(epochs):

    # Forward Propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    output = sigmoid(final_input)

    # Compute Loss (Mean Squared Error)
    loss = np.mean((y - output) ** 2)
    losses.append(loss)

    # Backpropagation
    d_output = (y - output) * sigmoid_derivative(output)
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(hidden_output)

    # Update Weights
    W2 += np.dot(hidden_output.T, d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    W1 += np.dot(X.T, d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# 6. Final Predictions
final_predictions = np.round(output)

print("------ Hyper-Parameter Tuning Result ------")
print("Learning Rate :", learning_rate)
print("Hidden Neurons:", hidden_neurons)
print("Epochs        :", epochs)

print("\nFinal Predictions:")
print(final_predictions)

print("\nFinal Loss:", loss)

# 7. Plot Training Loss Graph
plt.figure()
plt.plot(losses)
plt.title("Training Loss vs Epochs (BPNN - XOR)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()