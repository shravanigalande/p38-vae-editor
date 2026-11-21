"""U2 comparison: a small conditional GAN (cGAN) on Fashion-MNIST.

VAE edit  = encode a REAL image, shift its latent code, decode  (keeps the item's identity)
cGAN edit = keep the same noise vector, swap the class label    (cannot take a real image as input)
evaluate.py compares both with CLIP.

Usage: python train_gan.py --epochs 30
"""
import argparse, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from common import *

NOISE = 64


def build_generator():
    z = keras.Input((NOISE,)); y = keras.Input((1,), dtype="int32")
    e = layers.Flatten()(layers.Embedding(10, 32)(y))
    x = layers.Concatenate()([z, e])
    x = layers.Dense(7 * 7 * 128)(x); x = layers.LeakyReLU(0.2)(x); x = layers.Reshape((7, 7, 128))(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(32, 4, strides=2, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    out = layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    return keras.Model([z, y], out, name="generator")


def build_discriminator():
    img = keras.Input((28, 28, 1)); y = keras.Input((1,), dtype="int32")
    e = layers.Flatten()(layers.Embedding(10, 28 * 28)(y)); e = layers.Reshape((28, 28, 1))(e)
    x = layers.Concatenate()([img, e])
    x = layers.Conv2D(64, 4, strides=2, padding="same")(x); x = layers.LeakyReLU(0.2)(x); x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, 4, strides=2, padding="same")(x); x = layers.LeakyReLU(0.2)(x); x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    return keras.Model([img, y], layers.Dense(1)(x), name="discriminator")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    xtr, ytr, _, _ = load_fashion_mnist()
    G, D = build_generator(), build_discriminator()
    g_opt = keras.optimizers.Adam(2e-4, 0.5); d_opt = keras.optimizers.Adam(2e-4, 0.5)
    bce = keras.losses.BinaryCrossentropy(from_logits=True)
    ds = tf.data.Dataset.from_tensor_slices((xtr, ytr.astype("int32"))).shuffle(10000).batch(a.batch, drop_remainder=True)

    @tf.function
    def step(x, y):
        n = tf.shape(x)[0]
        z = tf.random.normal((n, NOISE))
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake = G([z, y[:, None]], training=True)
            d_real = D([x, y[:, None]], training=True)
            d_fake = D([fake, y[:, None]], training=True)
            d_loss = bce(tf.ones_like(d_real) * 0.9, d_real) + bce(tf.zeros_like(d_fake), d_fake)
            g_loss = bce(tf.ones_like(d_fake), d_fake)
        d_opt.apply_gradients(zip(dt.gradient(d_loss, D.trainable_variables), D.trainable_variables))
        g_opt.apply_gradients(zip(gt.gradient(g_loss, G.trainable_variables), G.trainable_variables))
        return d_loss, g_loss

    for ep in range(a.epochs):
        ls = np.mean([[float(v) for v in step(x, y)] for x, y in ds], axis=0)
        print(f"epoch {ep+1:02d}  D {ls[0]:.3f}  G {ls[1]:.3f}")
    G.save(os.path.join(MODELS, "cgan_generator.keras"))


if __name__ == "__main__":
    main()
