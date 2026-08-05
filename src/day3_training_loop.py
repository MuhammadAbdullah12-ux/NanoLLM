import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducible results
torch.manual_seed(42)

# 1. Define a simple 2-layer Neural Network
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Layer 1
        self.relu = nn.ReLU()                       # Activation function
        self.fc2 = nn.Linear(hidden_dim, output_dim) # Layer 2

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 2. Create synthetic dataset (100 samples, 10 features each)
# Target equation: target = 2 * x_0 + 3 * x_1 (simple linear relationship)
X = torch.randn(100, 10)
y = (2 * X[:, 0] + 3 * X[:, 1]).unsqueeze(1)

# 3. Instantiate model, loss function, and optimizer
model = SimpleNet(input_dim=10, hidden_dim=32, output_dim=1)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

print("Starting training...\n")

# 4. The Training Loop (100 epochs)
for epoch in range(1, 101):
    # Step 1: Clear old gradients
    optimizer.zero_grad()

    # Step 2: Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)

    # Step 3: Backward pass
    loss.backward()

    # Step 4: Update weights
    optimizer.step()

    # Log progress every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:3d}/100] | Loss: {loss.item():.6f}")

print("\nTraining completed successfully!")
