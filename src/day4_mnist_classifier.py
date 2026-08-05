import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Set random seed for reproducible results
torch.manual_seed(42)

# 2. Define Data Transformations (Convert raw images to Tensors & Normalize pixel values)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST dataset mean and standard deviation
])

print("Downloading and loading MNIST dataset...")
train_dataset = datasets.MNIST(root='./data/raw', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data/raw', train=False, download=True, transform=transform)

# DataLoader automatically splits dataset into mini-batches of 64 images
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 3. Define Neural Network for Digit Classification (28x28 pixels = 784 inputs -> 10 output digit classes)
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # Reshape 28x28 2D image matrix into 784 1D vector
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10) # 10 outputs corresponding to digit scores 0 through 9
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits

# 4. Instantiate Model, Loss Function (CrossEntropyLoss), and Optimizer (Adam)
model = MNISTClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop for 3 Epochs
epochs = 3
print("\nStarting training on MNIST...\n")

for epoch in range(1, epochs + 1):
    model.train() # Set model to training mode
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Step 1: Clear old gradients
        optimizer.zero_grad()

        # Step 2: Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Step 3: Backward pass
        loss.backward()

        # Step 4: Update weights
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_loss:.4f}")

# 6. Evaluation / Testing Phase
model.eval() # Set model to evaluation mode
correct = 0
total = 0

with torch.no_grad(): # Disable gradient calculation for faster evaluation
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1) # Pick digit class with highest score
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = (correct / total) * 100
print(f"\nFinal Test Accuracy: {accuracy:.2f}%")
