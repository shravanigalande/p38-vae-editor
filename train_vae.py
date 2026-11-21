"""U4 + U1: train a convolutional VAE in TensorFlow 2, then compute class centroids.

Usage:  python train_vae.py --epochs 30
Outputs: models/encoder.keras, models/decoder.keras, models/centroids.npy,
         assets/test_subset.npz, results/vae_loss.png, results/latent_pca.png
"""
import argparse, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *


def build_encoder(latent=LATENT_DIM):
    inp = keras.Input((28, 28, 1))
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inp)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    mu = layers.Dense(latent, name="z_mean")(x)
    logvar = layers.Dense(latent, name="z_log_var")(x)
    return keras.Model(inp, [mu, logvar], name="encoder")


def build_decoder(latent=LATENT_DIM):
    inp = keras.Input((latent,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(inp)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    out = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)
    return keras.Model(inp, out, name="decoder")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--beta", type=float, default=1.0, help="KL weight")
    a = ap.parse_args()

    xtr, ytr, xte, yte = load_fashion_mnist()
    enc, dec = build_encoder(), build_decoder()
    opt = keras.optimizers.Adam(1e-3)
    ds = tf.data.Dataset.from_tensor_slices(xtr).shuffle(10000).batch(a.batch)

    @tf.function
    def step(x):
        with tf.GradientTape() as tape:
            mu, logvar = enc(x, training=True)
            eps = tf.random.normal(tf.shape(mu))
            z = mu + tf.exp(0.5 * logvar) * eps          # reparameterisation trick
            xr = dec(z, training=True)
            rec = tf.reduce_sum(keras.losses.binary_crossentropy(x, xr), axis=[1, 2])
            kl = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
            loss = tf.reduce_mean(rec + a.beta * kl)
        vars_ = enc.trainable_variables + dec.trainable_variables
        opt.apply_gradients(zip(tape.gradient(loss, vars_), vars_))
        return loss, tf.reduce_mean(rec), tf.reduce_mean(kl)

    hist = []
    for ep in range(a.epochs):
        ls = []
        for xb in ds:
            ls.append([float(v) for v in step(xb)])
        m = np.mean(ls, axis=0)
        hist.append(m)
        print(f"epoch {ep+1:02d}  loss {m[0]:.2f}  recon {m[1]:.2f}  kl {m[2]:.2f}")

    enc.save(os.path.join(MODELS, "encoder.keras"))
    dec.save(os.path.join(MODELS, "decoder.keras"))

    # ---- U1: class centroids in latent space (using the posterior mean) ----
    mu_tr, _ = enc.predict(xtr, batch_size=512, verbose=0)
    centroids = np.stack([mu_tr[ytr == c].mean(0) for c in range(10)])
    np.save(os.path.join(MODELS, "centroids.npy"), centroids)

    # small test subset for the app (100 per class)
    idx = np.concatenate([np.where(yte == c)[0][:100] for c in range(10)])
    np.savez_compressed(os.path.join(ASSETS, "test_subset.npz"), x=xte[idx], y=yte[idx])

    # ---- plots ----
    h = np.array(hist)
    plt.figure(figsize=(5, 3)); plt.plot(h[:, 0], label="total"); plt.plot(h[:, 1], label="recon")
    plt.plot(h[:, 2], label="KL"); plt.xlabel("epoch"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "vae_loss.png"), dpi=150); plt.close()

    from sklearn.decomposition import PCA
    pca = PCA(2).fit(mu_tr)
    p = pca.transform(mu_tr[:5000]); pc = pca.transform(centroids)
    plt.figure(figsize=(6, 5))
    for c in range(10):
        m = ytr[:5000] == c
        plt.scatter(p[m, 0], p[m, 1], s=3, alpha=0.4, label=CLASS_NAMES[c])
        plt.scatter(*pc[c], marker="X", s=140, edgecolor="k")
        plt.annotate(CLASS_NAMES[c], pc[c], fontsize=8)
    plt.title("Latent space (PCA of 16-D) with class centroids (X)")
    plt.legend(markerscale=3, fontsize=6); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "latent_pca.png"), dpi=150); plt.close()

    # reconstruction MSE on test set (reported in the model card)
    mu_te, _ = enc.predict(xte, batch_size=512, verbose=0)
    rec = dec.predict(mu_te, batch_size=512, verbose=0)
    mse = float(np.mean((rec - xte) ** 2))
    with open(os.path.join(RESULTS, "vae_metrics.txt"), "w") as f:
        f.write(f"test_recon_mse={mse:.5f}\nfinal_loss={h[-1,0]:.2f}\nepochs={a.epochs}\n")
    print("test reconstruction MSE:", mse)


if __name__ == "__main__":
    main()
