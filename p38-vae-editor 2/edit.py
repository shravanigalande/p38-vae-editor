"""Latent-space editing logic (U1/U2): move a latent code toward a target class centroid."""
import os
import numpy as np
from common import *

_cache = {}


def _load():
    if not _cache:
        from tensorflow import keras
        _cache["enc"] = keras.models.load_model(os.path.join(MODELS, "encoder.keras"), compile=False)
        _cache["dec"] = keras.models.load_model(os.path.join(MODELS, "decoder.keras"), compile=False)
        _cache["cent"] = np.load(os.path.join(MODELS, "centroids.npy"))
    return _cache["enc"], _cache["dec"], _cache["cent"]


def encode(img):
    enc, _, _ = _load()
    mu, _ = enc.predict(img[None], verbose=0)
    return mu[0]


def decode(z):
    _, dec, _ = _load()
    z = np.atleast_2d(z)
    return dec.predict(z, verbose=0)[..., 0]


def nearest_class(z):
    _, _, cent = _load()
    return int(np.argmin(np.linalg.norm(cent - z, axis=1)))


def edit_latent(z, target, alpha, source=None):
    """z' = z + alpha * (centroid[target] - centroid[source]).

    Using the *difference of centroids* (not "jump to the centroid") keeps what is
    specific to this item (its shape/brightness) while shifting its class.
    alpha=0 -> unchanged, alpha=1 -> full class shift, alpha>1 -> exaggerated.
    """
    _, _, cent = _load()
    if source is None:
        source = nearest_class(z)
    return z + alpha * (cent[target] - cent[source])


def morph_strip(img, target, max_alpha=1.0, steps=9):
    """Returns (N,28,28) images for alpha = 0 .. max_alpha."""
    z = encode(img)
    alphas = np.linspace(0, max_alpha, steps)
    zs = np.stack([edit_latent(z, target, a) for a in alphas])
    return decode(zs), alphas
