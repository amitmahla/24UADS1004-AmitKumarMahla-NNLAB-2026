# ---------------------------------------
# BPNN USING TENSORFLOW (XOR PROBLEM)
# ---------------------------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Dataset (XOR)
# -------------------------------
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

print("\n==============================")
print("   BPNN USING TENSORFLOW")
print("==============================")

print("\nDataset:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Output: {y[i][0]}")

# -------------------------------
# Step 2: Create Model
# -------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_dim=2, activation='relu'),  # Hidden Layer
    tf.keras.layers.Dense(1, activation='sigmoid')             # Output Layer
])

# -------------------------------
# Step 3: Compile Model
# -------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Compiled Successfully!")

# -------------------------------
# Step 4: Train Model
# -------------------------------
print("\nTraining Started...\n")

history = model.fit(X, y, epochs=500, verbose=0)

print("Training Completed!")

# -------------------------------
# Step 5: Predictions
# -------------------------------
predictions = model.predict(X)

print("\n==============================")
print("   PREDICTED OUTPUTS")
print("==============================")

for i in range(len(X)):
    print(f"Input: {X[i]} -> Output: {predictions[i][0]:.4f}")

# -------------------------------
# Step 6: Accuracy
# -------------------------------
pred_binary = (predictions > 0.5).astype(int)
accuracy = np.mean(pred_binary == y) * 100

print("\n==============================")
print("   MODEL PERFORMANCE")
print("==============================")
print(f"Accuracy: {accuracy:.2f}%")

# -------------------------------
# Step 7: Plot Loss Graph
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'])
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()