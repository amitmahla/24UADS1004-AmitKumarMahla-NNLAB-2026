# Import libraries
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Step 1: Load Dataset (Dummy Medical Images)
# -------------------------------
# Using CIFAR10 as placeholder for medical dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to binary (for simplicity)
y_train = (y_train > 4).astype(int)
y_test = (y_test > 4).astype(int)

# -------------------------------
# Step 2: Load Pre-trained Model
# -------------------------------
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# -------------------------------
# Step 3: Add Custom Layers
# -------------------------------
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# -------------------------------
# Step 4: Compile Model
# -------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Step 5: Train Model
# -------------------------------
history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

# -------------------------------
# Step 6: Evaluate Model
# -------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", test_acc)

# -------------------------------
# Step 7: Plot Graph
# -------------------------------
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()