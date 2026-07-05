import os, glob
import numpy as np
from PIL import Image
import time

def load_split(root, split, n_per_class=None, img_size=28):
    X = []
    t0 = time.time()
    for digit in range(10):
        folder = os.path.join(root, split, str(digit))
        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        if n_per_class is not None:
            files = files[:n_per_class]
        for f in files:
            img = Image.open(f).convert("L")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            X.append(arr.flatten())
    X = np.array(X, dtype=np.float32)
    print(f"Loaded {split}: {X.shape} in {time.time()-t0:.1f}s")
    return X

if __name__ == "__main__":
    root = "/home/claude/data/mnist_png"
    X_train = load_split(root, "training", n_per_class=1500)   # 15,000 images
    X_test  = load_split(root, "testing", n_per_class=300)     # 3,000 images
    np.save("/home/claude/X_train.npy", X_train)
    np.save("/home/claude/X_test.npy", X_test)
    print("Saved arrays.")
