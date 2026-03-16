# Import required libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load MNIST Dataset
# -------------------------------
# MNIST contains handwritten digit images (0–9)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# -------------------------------
# Step 2: Data Preprocessing
# -------------------------------
# Reshape data to include channel (28x28x1 for grayscale images)
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0   # Normalize pixel values (0–1)
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# -------------------------------
# Step 3: Build CNN Model
# -------------------------------
model = keras.Sequential([

    # First Convolution Layer (Feature Extraction)
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),

    # Pooling Layer (Reduce dimensions)
    layers.MaxPooling2D((2,2)),

    # Second Convolution Layer (Deeper features)
    layers.Conv2D(64, (3,3), activation='relu'),

    # Second Pooling Layer
    layers.MaxPooling2D((2,2)),

    # Flatten layer (Convert 2D to 1D)
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(64, activation='relu'),

    # Output Layer (10 classes → digits 0–9)
    layers.Dense(10, activation='softmax')
])

# -------------------------------
# Step 4: Compile Model
# -------------------------------
# Adam optimizer + categorical loss
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Compiled Successfully!")

# -------------------------------
# Step 5: Train Model
# -------------------------------
# Train for 5 epochs
history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

# -------------------------------
# Step 6: Evaluate Model
# -------------------------------
# Test accuracy on unseen data
test_loss, test_acc = model.evaluate(x_test, y_test)

print("\n==============================")
print("   MODEL PERFORMANCE")
print("==============================")
print("Test Accuracy:", test_acc)

# -------------------------------
# Step 7: Plot Loss Graph
# -------------------------------
# Visualize training & validation loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()