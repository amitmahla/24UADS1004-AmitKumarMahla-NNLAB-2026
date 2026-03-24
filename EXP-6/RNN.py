# Import libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Create Dummy Sequence Dataset
# -------------------------------
# Example: Predict next number in sequence
X = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6]], dtype=np.float32)

y = np.array([[4],
              [5],
              [6],
              [7]], dtype=np.float32)

# Convert to tensors
X = torch.from_numpy(X).unsqueeze(-1)  # shape: (batch, seq_len, input_size)
y = torch.from_numpy(y)

# -------------------------------
# Step 2: Define RNN Model
# -------------------------------
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=8, batch_first=True)
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        out, _ = self.rnn(x)             # RNN output
        out = out[:, -1, :]              # Take last time step
        out = self.fc(out)               # Fully connected layer
        return out

model = RNNModel()

# -------------------------------
# Step 3: Loss and Optimizer
# -------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------------
# Step 4: Training
# -------------------------------
loss_list = []

for epoch in range(100):
    output = model(X)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())

    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

# -------------------------------
# Step 5: Predictions
# -------------------------------
with torch.no_grad():
    predictions = model(X)

print("\nPredictions:")
for i in range(len(X)):
    print(f"Input: {X[i].squeeze().numpy()} -> Output: {predictions[i].item():.2f}")

# -------------------------------
# Step 6: Plot Loss Graph
# -------------------------------
plt.plot(loss_list)
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()