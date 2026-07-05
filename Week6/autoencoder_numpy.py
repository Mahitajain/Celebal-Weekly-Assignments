"""
Denoising Autoencoder for MNIST - built from scratch with NumPy
(no torch/tensorflow available in this environment, so forward/backward
 pass and Adam optimizer are implemented manually)

Architecture (mirrors the "FFNN Autoencoder" design from the reference
GitHub repo, but trained to map NOISY -> CLEAN images instead of
clean -> clean):

    Input (784) -> Dense(256, ReLU) -> Dense(64, ReLU)   [encoder]
    -> Dense(256, ReLU) -> Dense(784, Sigmoid)            [decoder]
"""
import numpy as np
import time

rng = np.random.default_rng(42)

def add_noise(X, noise_factor=0.4):
    noisy = X + noise_factor * rng.standard_normal(X.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(np.float32)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

class DenoisingAutoencoder:
    def __init__(self, layer_sizes=(784, 256, 64, 256, 784), seed=0):
        r = np.random.default_rng(seed)
        self.sizes = layer_sizes
        self.W, self.b = [], []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            std = 1.0 / np.sqrt(fan_in)  # same init scheme as reference repo
            self.W.append(r.normal(0, std, size=(layer_sizes[i], layer_sizes[i+1])).astype(np.float32))
            self.b.append(np.zeros(layer_sizes[i+1], dtype=np.float32))
        # Adam optimizer state
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(bb) for bb in self.b]
        self.vb = [np.zeros_like(bb) for bb in self.b]
        self.t = 0

    def forward(self, X):
        acts = [X]
        zs = []
        n_layers = len(self.W)
        for i in range(n_layers):
            z = acts[-1] @ self.W[i] + self.b[i]
            zs.append(z)
            if i < n_layers - 1:
                acts.append(relu(z))
            else:
                acts.append(sigmoid(z))  # output layer
        return acts, zs

    def backward(self, acts, zs, y_true):
        n = y_true.shape[0]
        n_layers = len(self.W)
        grads_W = [None] * n_layers
        grads_b = [None] * n_layers

        # MSE loss: dL/dyhat = 2*(yhat - y)/n ; combined with sigmoid derivative
        yhat = acts[-1]
        dz = (2.0 / n) * (yhat - y_true) * yhat * (1 - yhat)  # sigmoid output layer

        for i in reversed(range(n_layers)):
            a_prev = acts[i]
            grads_W[i] = a_prev.T @ dz
            grads_b[i] = dz.sum(axis=0)
            if i > 0:
                da_prev = dz @ self.W[i].T
                dz = da_prev * relu_deriv(zs[i-1])
        return grads_W, grads_b

    def adam_step(self, grads_W, grads_b, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for i in range(len(self.W)):
            self.mW[i] = beta1*self.mW[i] + (1-beta1)*grads_W[i]
            self.vW[i] = beta2*self.vW[i] + (1-beta2)*(grads_W[i]**2)
            mW_hat = self.mW[i] / (1 - beta1**self.t)
            vW_hat = self.vW[i] / (1 - beta2**self.t)
            self.W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

            self.mb[i] = beta1*self.mb[i] + (1-beta1)*grads_b[i]
            self.vb[i] = beta2*self.vb[i] + (1-beta2)*(grads_b[i]**2)
            mb_hat = self.mb[i] / (1 - beta1**self.t)
            vb_hat = self.vb[i] / (1 - beta2**self.t)
            self.b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def predict(self, X):
        acts, _ = self.forward(X)
        return acts[-1]

def mse(a, b):
    return float(np.mean((a - b) ** 2))

def train(model, X_train_clean, X_val_clean, epochs=25, batch_size=128, lr=1e-3, noise_factor=0.4):
    n = X_train_clean.shape[0]
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        perm = rng.permutation(n)
        X_shuf = X_train_clean[perm]
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            clean_batch = X_shuf[start:start+batch_size]
            noisy_batch = add_noise(clean_batch, noise_factor)
            acts, zs = model.forward(noisy_batch)
            loss = mse(acts[-1], clean_batch)
            epoch_loss += loss
            n_batches += 1
            grads_W, grads_b = model.backward(acts, zs, clean_batch)
            model.adam_step(grads_W, grads_b, lr=lr)
        train_loss = epoch_loss / n_batches

        # validation
        noisy_val = add_noise(X_val_clean, noise_factor)
        val_pred = model.predict(noisy_val)
        val_loss = mse(val_pred, X_val_clean)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:2d}/{epochs} - train_mse: {train_loss:.5f} - val_mse: {val_loss:.5f} - {time.time()-t0:.1f}s")
    return history

if __name__ == "__main__":
    X_train = np.load("/home/claude/X_train.npy")
    X_test  = np.load("/home/claude/X_test.npy")

    # split a validation set out of training data
    n_val = 1500
    X_val = X_train[:n_val]
    X_tr  = X_train[n_val:]

    model = DenoisingAutoencoder(layer_sizes=(784, 256, 64, 256, 784), seed=42)
    history = train(model, X_tr, X_val, epochs=25, batch_size=128, lr=1e-3, noise_factor=0.4)

    # Final test evaluation
    noisy_test = add_noise(X_test, 0.4)
    test_pred = model.predict(noisy_test)
    test_mse = mse(test_pred, X_test)
    print(f"\nFinal TEST mse: {test_mse:.5f}")

    np.savez("/home/claude/autoencoder_weights.npz",
             *[w for w in model.W], *[b for b in model.b])
    np.save("/home/claude/history.npy", history, allow_pickle=True)
    print("Saved weights and history.")
