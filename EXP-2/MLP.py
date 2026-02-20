import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# -----------------------------
# ACTIVATION FUNCTIONS
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


# -----------------------------
# DATASET (XOR PROBLEM)
# -----------------------------
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

print("Training Dataset (X):")
print(X)

print("\nTarget Output (y):")
print(y)


# -----------------------------
# INITIALIZATION
# -----------------------------
np.random.seed(42)

input_neurons = 2
hidden_neurons = 4
output_neurons = 1

learning_rate = 0.1
epochs = 10000

# Weight and Bias Initialization
wh = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
bh = np.random.uniform(-1, 1, (1, hidden_neurons))

wo = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
bo = np.random.uniform(-1, 1, (1, output_neurons))

print("\nInitial Hidden Layer Weights (wh):")
print(wh)

print("\nInitial Output Layer Weights (wo):")
print(wo)


# -----------------------------
# TRAINING PROCESS
# -----------------------------
loss_list = []

for epoch in range(epochs):

    # -------------------------
    # FORWARD PROPAGATION
    # -------------------------
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, wo) + bo
    predicted_output = sigmoid(final_input)

    # -------------------------
    # ERROR AND LOSS
    # -------------------------
    error = y - predicted_output
    loss = np.mean(np.square(error))
    loss_list.append(loss)

    # -------------------------
    # BACKPROPAGATION
    # -------------------------
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(wo.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)

    # -------------------------
    # WEIGHT UPDATION
    # -------------------------
    wo += hidden_output.T.dot(d_predicted_output) * learning_rate
    bo += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate

    wh += X.T.dot(d_hidden_layer) * learning_rate
    bh += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}  Loss: {loss:.6f}")


# -----------------------------
# FINAL PREDICTIONS
# -----------------------------
print("\n-----------------------------")
print("TRAINING COMPLETED")
print("-----------------------------")

print("\nFinal Predicted Output (Sigmoid Values):")
print(predicted_output)

binary_predictions = (predicted_output > 0.5).astype(int)

print("\nFinal Predicted Output (Binary 0/1):")
print(binary_predictions)


# -----------------------------
# OUTPUT TABLE
# -----------------------------
print("\n-----------------------------")
print("FINAL RESULT TABLE")
print("-----------------------------")
print("Input  | Expected | Predicted(Sigmoid) | Predicted(Binary)")
print("---------------------------------------------------------")

for i in range(len(X)):
    print(f"{X[i]}   |   {y[i][0]}      |     {predicted_output[i][0]:.4f}       |        {binary_predictions[i][0]}")


# -----------------------------
# ACCURACY
# -----------------------------
acc = accuracy_score(y, binary_predictions)
print("\nAccuracy:", acc * 100, "%")


# -----------------------------
# CONFUSION MATRIX
# -----------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y, binary_predictions))


# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\nClassification Report:")
print(classification_report(y, binary_predictions))


# -----------------------------
# CUSTOM TESTING
# -----------------------------
print("\n-----------------------------")
print("CUSTOM TESTING")
print("-----------------------------")

test_data = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])

hidden_test_input = np.dot(test_data, wh) + bh
hidden_test_output = sigmoid(hidden_test_input)

final_test_input = np.dot(hidden_test_output, wo) + bo
test_prediction = sigmoid(final_test_input)

binary_test_prediction = (test_prediction > 0.5).astype(int)

print("\nTest Inputs:")
print(test_data)

print("\nTest Prediction (Sigmoid):")
print(test_prediction)

print("\nTest Prediction (Binary):")
print(binary_test_prediction)


# -----------------------------
# LOSS GRAPH
# -----------------------------
plt.plot(loss_list)
plt.title("Loss vs Epochs (MLP Training)")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error Loss")
plt.grid(True)
plt.show()
