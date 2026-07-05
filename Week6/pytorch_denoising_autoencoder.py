"""
Convolutional Denoising Autoencoder for MNIST (PyTorch)
========================================================
Run this on your own machine (with torch + torchvision installed and,
ideally, a GPU). It mirrors the "Upsampled CNN Autoencoder" design from
https://github.com/NvsYashwanth/MNIST-Autoecncoder, adapted for the
DENOISING task: the model is fed a noisy image and trained to output
the corresponding clean image.

Install once:
    pip install torch torchvision matplotlib

Usage:
    python pytorch_denoising_autoencoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------------------------------------
# Data: MNIST via torchvision (downloads automatically). If you'd rather
# use the mnist_png folder you already have, swap this for
# torchvision.datasets.ImageFolder pointed at mnist_png/training etc.
# ---------------------------------------------------------------------
transform = transforms.ToTensor()  # scales pixels to [0, 1]

train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

NOISE_FACTOR = 0.4

def add_noise(imgs, noise_factor=NOISE_FACTOR):
    noisy = imgs + noise_factor * torch.randn_like(imgs)
    return torch.clamp(noisy, 0.0, 1.0)

# ---------------------------------------------------------------------
# Model: Conv encoder -> compressed 7x7x8 representation -> upsample
# + conv decoder (avoids checkerboard artifacts vs. transpose conv)
# ---------------------------------------------------------------------
class ConvDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, 3, padding=1)   # 28x28x32
        self.enc_conv2 = nn.Conv2d(32, 16, 3, padding=1)  # 14x14x16
        self.enc_conv3 = nn.Conv2d(16, 8, 3, padding=1)   # 7x7x8
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder (upsample + conv, instead of transpose conv, to avoid
        # checkerboard artifacts)
        self.dec_conv1 = nn.Conv2d(8, 16, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.dec_conv3 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        # Encode
        x = F.relu(self.enc_conv1(x))
        x = self.pool(x)                 # 14x14x32
        x = F.relu(self.enc_conv2(x))
        x = self.pool(x)                 # 7x7x16
        x = F.relu(self.enc_conv3(x))    # 7x7x8  <- compressed representation

        # Decode
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # 14x14x8
        x = F.relu(self.dec_conv1(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # 28x28x16
        x = F.relu(self.dec_conv2(x))
        x = torch.sigmoid(self.dec_conv3(x))                  # 28x28x1
        return x

model = ConvDenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 20

def train():
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, images)  # compare to CLEAN images
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{EPOCHS} - train MSE: {epoch_loss:.5f}")

def evaluate_and_plot():
    model.eval()
    images, _ = next(iter(test_loader))
    images = images.to(device)
    noisy_images = add_noise(images)
    with torch.no_grad():
        outputs = model(noisy_images)

    images_np = images.cpu().numpy()
    noisy_np = noisy_images.cpu().numpy()
    outputs_np = outputs.cpu().numpy()

    n = 10
    fig, axes = plt.subplots(3, n, figsize=(16, 5))
    for i in range(n):
        axes[0, i].imshow(images_np[i, 0], cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(noisy_np[i, 0], cmap="gray"); axes[1, i].axis("off")
        axes[2, i].imshow(outputs_np[i, 0], cmap="gray"); axes[2, i].axis("off")
    axes[0, 0].set_title("Original", loc="left")
    axes[1, 0].set_title("Noisy", loc="left")
    axes[2, 0].set_title("Denoised", loc="left")
    plt.tight_layout()
    plt.savefig("pytorch_denoising_results.png", dpi=150)
    print("Saved pytorch_denoising_results.png")

if __name__ == "__main__":
    train()
    evaluate_and_plot()
    torch.save(model.state_dict(), "conv_denoising_autoencoder.pth")
    print("Saved model weights to conv_denoising_autoencoder.pth")
