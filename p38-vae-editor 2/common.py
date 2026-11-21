"""Shared helpers: class names, data loading, paths."""
import os
import numpy as np

CLASS_NAMES = ["t-shirt", "trouser", "pullover", "dress", "coat",
               "sandal", "shirt", "sneaker", "bag", "ankle boot"]
LATENT_DIM = 16

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")
ASSETS = os.path.join(ROOT, "assets")
RESULTS = os.path.join(ROOT, "results")
for _d in (MODELS, ASSETS, RESULTS, os.path.join(RESULTS, "strips")):
    os.makedirs(_d, exist_ok=True)


def load_fashion_mnist():
    """Returns (x_train, y_train, x_test, y_test) with x in [0,1], shape (N,28,28,1).

    Set SMOKE=1 to use tiny random data (only for testing the code without internet).
    """
    if os.environ.get("SMOKE") == "1":
        rng = np.random.default_rng(0)
        xt = rng.random((600, 28, 28, 1), dtype=np.float32)
        yt = np.arange(600) % 10
        return xt, yt, xt[:200], yt[:200]
    local = os.environ.get("FASHION_DIR")
    if local:  # raw idx .gz files (used when keras can't download)
        import gzip
        def rd(name, img):
            with gzip.open(os.path.join(local, name), "rb") as f:
                return np.frombuffer(f.read(), np.uint8, offset=16 if img else 8)
        xtr = rd("train-images-idx3-ubyte.gz", True).reshape(-1, 28, 28)
        ytr = rd("train-labels-idx1-ubyte.gz", False)
        xte = rd("t10k-images-idx3-ubyte.gz", True).reshape(-1, 28, 28)
        yte = rd("t10k-labels-idx1-ubyte.gz", False)
    else:
        import tensorflow as tf
        (xtr, ytr), (xte, yte) = tf.keras.datasets.fashion_mnist.load_data()
    f = lambda x: (x.astype("float32") / 255.0)[..., None]
    return f(xtr), ytr, f(xte), yte


def load_test_subset():
    """Small test-image subset shipped in the repo so the app needs no download."""
    p = os.path.join(ASSETS, "test_subset.npz")
    d = np.load(p)
    return d["x"], d["y"]
