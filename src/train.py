import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import ImageDeblurDataset
import os

# Define Student CNN
class StudentCNN(nn.Module):
    def __init__(self):
        super(StudentCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Paths
blurry_path = "data/blurry"
teacher_path = "data/teacher_output"

# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Dataset and Dataloader
dataset = ImageDeblurDataset(blurry_path, teacher_path, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model, Loss, Optimizer
model = StudentCNN().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for i, (blurry, teacher) in enumerate(dataloader):
        blurry = blurry.to(device)
        teacher = teacher.to(device)

        output = model(blurry)
        loss = criterion(output, teacher)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Save model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/student_model.pth")
print("Model saved to checkpoints/student_model.pth")
  