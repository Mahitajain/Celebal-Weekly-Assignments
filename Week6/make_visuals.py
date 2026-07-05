import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from autoencoder_numpy import DenoisingAutoencoder, add_noise, mse

X_test = np.load("/home/claude/X_test.npy")

# rebuild model and load weights
model = DenoisingAutoencoder(layer_sizes=(784, 256, 64, 256, 784), seed=42)
data = np.load("/home/claude/autoencoder_weights.npz")
arrs = [data[f"arr_{i}"] for i in range(len(data.files))]
n_layers = len(model.W)
model.W = arrs[:n_layers]
model.b = arrs[n_layers:]

rng = np.random.default_rng(123)
idx = rng.choice(len(X_test), size=10, replace=False)
clean = X_test[idx]
noisy = add_noise(clean, noise_factor=0.4)
denoised = model.predict(noisy)

fig, axes = plt.subplots(3, 10, figsize=(16, 5))
row_labels = ["Original", "Noisy Input", "Denoised (model output)"]
for col in range(10):
    for row, imgs in enumerate([clean, noisy, denoised]):
        ax = axes[row, col]
        ax.imshow(imgs[col].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(row_labels[row])
for row in range(3):
    axes[row, 0].text(-10, 14, row_labels[row], rotation=90, va="center", ha="center", fontsize=11)

plt.suptitle("MNIST Denoising Autoencoder — Original vs Noisy vs Reconstructed", fontsize=14)
plt.tight_layout()
plt.savefig("/home/claude/denoising_results.png", dpi=150, bbox_inches="tight")
print("Saved denoising_results.png")

# training curve
history = np.load("/home/claude/history.npy", allow_pickle=True).item()
plt.figure(figsize=(6,4))
plt.plot(history["train_loss"], label="Train MSE")
plt.plot(history["val_loss"], label="Val MSE")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.title("Training Curve")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/claude/training_curve.png", dpi=150)
print("Saved training_curve.png")

print(f"\nSample-level MSE: noisy={mse(noisy, clean):.5f}  denoised={mse(denoised, clean):.5f}")
