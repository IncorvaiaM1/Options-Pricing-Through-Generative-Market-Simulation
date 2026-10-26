"""Return discretization for the tokenized-AR baseline.

scheme="quantile": equal-mass bins fit on training returns (common in practice; few bins in the tails).
scheme="uniform":  equal-width bins on [-clip, clip] (Chronos-style); everything beyond clip is clamped.
Decoding modes: "mean" (mean of training data in the bin; default, fairest for the wide edge bins),
"center" (bin midpoint), "dequant" (uniform jitter inside the bin).
"""
import jax
import jax.numpy as jnp
import numpy as np


def fit_bins(train_norm, n_bins, scheme="quantile", clip=6.0):
    if scheme == "quantile":
        edges = np.quantile(train_norm.ravel(), np.linspace(0, 1, n_bins + 1))
    elif scheme == "uniform":
        edges = np.linspace(-clip, clip, n_bins + 1)
    else:
        raise ValueError(scheme)
    centers = 0.5 * (edges[1:] + edges[:-1])
    x = np.asarray(train_norm).ravel()
    idx = np.clip(np.searchsorted(edges[1:-1], x, side="right"), 0, n_bins - 1)
    sums = np.bincount(idx, weights=x, minlength=n_bins)
    cnts = np.bincount(idx, minlength=n_bins)
    means = np.where(cnts > 0, sums / np.maximum(cnts, 1), centers)
    return dict(edges=jnp.asarray(edges, jnp.float32), centers=jnp.asarray(centers, jnp.float32),
                means=jnp.asarray(means, jnp.float32))


def encode(x, bins):
    inner = bins["edges"][1:-1]
    return jnp.searchsorted(inner, x, side="right").astype(jnp.int32)  # values outside range clamp to edge bins


def decode(idx, bins, mode="mean", key=None):
    if mode == "mean":
        return bins["means"][idx]
    if mode == "center":
        return bins["centers"][idx]
    lo, hi = bins["edges"][idx], bins["edges"][idx + 1]
    return lo + (hi - lo) * jax.random.uniform(key, idx.shape)
