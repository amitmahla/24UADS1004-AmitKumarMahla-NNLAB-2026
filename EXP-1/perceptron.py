import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Generate Dataset
# -------------------------------
np.random.seed(42)

class1 = np.random.randn(50, 2) + np.array([2, 2])
class2 = np.random.randn(50, 2) + np.array([-2, -2])

X = np.vstack((class1, class2))
y = np.hstack((np.ones(50), -1*np.ones(50)))

# Add bias term
X_bias = np.hstack((np.ones((X.shape[0], 1)), X))

print("\n==============================")
print("   PERCEPTRON LEARNING ALGO")
print("==============================")

print("\nDataset Generated Successfully!")
print("Total Samples:", X.shape[0])
print("Features:", X.shape[1])
print("Class +1 Samples:", np.sum(y == 1))
print("Class -1 Samples:", np.sum(y == -1))

print("\nFirst 5 Training Samples:")
for i in range(5):
    print(f"X = {X[i]}, Label = {y[i]}")

# -------------------------------
# Step 2: Perceptron Training
# -------------------------------
def perceptron_train(X, y, lr=0.1, epochs=20):
    w = np.zeros(X.shape[1])
    errors = []

    print("\n==============================")
    print("   TRAINING STARTED")
    print("==============================")
    print("Initial Weights:", w)
    print("Learning Rate:", lr)
    print("Max Epochs:", epochs)

    for epoch in range(epochs):
        total_error = 0

        print(f"\n--- Epoch {epoch+1} ---")

        for i in range(len(X)):
            net = np.dot(X[i], w)
            prediction = np.sign(net)

            if prediction == 0:
                prediction = -1

            if prediction != y[i]:
                w_old = w.copy()
                w = w + lr * y[i] * X[i]
                total_error += 1

                print(f"Misclassified Sample {i}")
                print(f"Input: {X[i]} | Target: {y[i]} | Predicted: {prediction}")
                print("Old Weights:", w_old)
                print("New Weights:", w)
                print("---------------------------------")

        errors.append(total_error)

        print(f"Epoch {epoch+1} Completed -> Total Misclassifications: {total_error}")

        if total_error == 0:
            print("\n✅ Training Converged Successfully!")
            print(f"Converged at Epoch: {epoch+1}")
            break

    return w, errors

weights, errors = perceptron_train(X_bias, y, lr=0.1, epochs=20)

# -------------------------------
# Step 3: Final Training Output
# -------------------------------
print("\n==============================")
print("   TRAINING FINISHED")
print("==============================")
print("Final Weights:", weights)

# -------------------------------
# Step 4: Accuracy Calculation
# -------------------------------
predictions = np.sign(np.dot(X_bias, weights))
predictions[predictions == 0] = -1

accuracy = np.mean(predictions == y) * 100

print("\n==============================")
print("   MODEL PERFORMANCE")
print("==============================")
print("Accuracy:", accuracy, "%")

# Confusion Matrix
tp = np.sum((predictions == 1) & (y == 1))
tn = np.sum((predictions == -1) & (y == -1))
fp = np.sum((predictions == 1) & (y == -1))
fn = np.sum((predictions == -1) & (y == 1))

print("\nConfusion Matrix:")
print("TP:", tp, "  FP:", fp)
print("FN:", fn, "  TN:", tn)

# Sample Predictions
print("\nSample Predictions (First 10):")
for i in range(10):
    print(f"Input: {X[i]} -> Predicted: {predictions[i]} | Actual: {y[i]}")

# -------------------------------
# Step 5: Plot Decision Boundary
# -------------------------------
def plot_decision_boundary(X, y, w):
    plt.figure(figsize=(8, 6))

    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], color="blue", label="Class +1")
    plt.scatter(X[y == -1][:, 1], X[y == -1][:, 2], color="red", label="Class -1")

    x_values = np.linspace(-6, 6, 200)

    if w[2] != 0:
        y_values = -(w[0] + w[1] * x_values) / w[2]
        plt.plot(x_values, y_values, color="green", linewidth=2, label="Decision Boundary")

    plt.title("Perceptron Learning Algorithm - Decision Boundary")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_decision_boundary(X_bias, y, weights)

# -------------------------------
# Step 6: Plot Error Graph
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(errors, marker="o")
plt.title("Training Error vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Misclassifications")
plt.grid(True)
plt.show()

print("\n==============================")
print("   PROGRAM COMPLETED SUCCESSFULLY")
print("==============================")
