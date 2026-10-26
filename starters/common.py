"""Small shared helpers for the starters (JAX/Flax)."""
import json
import math
import os
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------- Track A data
def load_trackA(data_dir):
    cfg = json.load(open(os.path.join(data_dir, "config.json")))
    s = cfg["scale"]
    train = np.load(os.path.join(data_dir, "train.npy")) / s
    val = np.load(os.path.join(data_dir, "val.npy")) / s
    ctx = np.load(os.path.join(data_dir, "eval_ctx.npy")) / s
    return cfg, train.astype(np.float32), val.astype(np.float32), ctx.astype(np.float32)


def ctx_features(ctx):
    """ctx [B, L] normalized returns -> [B, L + 3]: signed-log returns + log realized var (5, 20, L days)."""
    f = jnp.sign(ctx) * jnp.log1p(jnp.abs(ctx))
    rv = [jnp.log(jnp.mean(ctx[:, -w:] ** 2, axis=1, keepdims=True) + 1e-4) for w in (5, 20, ctx.shape[1])]
    return jnp.concatenate([f] + rv, axis=1)


def write_samples(out_dir, fut_raw, meta):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "samples.npy"), np.asarray(fut_raw, np.float32))
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"wrote {out_dir}/samples.npy {np.shape(fut_raw)}", flush=True)


def n_params(params):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return time.time() - self.t0


# ---------------------------------------------------------------- layers
class MLP(nn.Module):
    widths: tuple
    act: callable = nn.silu
    final_act: bool = False

    @nn.compact
    def __call__(self, x):
        for i, w in enumerate(self.widths):
            x = nn.Dense(w)(x)
            if i < len(self.widths) - 1 or self.final_act:
                x = self.act(x)
        return x


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(half) / half)
    a = t.astype(jnp.float32)[..., None] * freqs
    return jnp.concatenate([jnp.cos(a), jnp.sin(a)], -1)


class AdaLNBlock(nn.Module):
    w: int

    @nn.compact
    def __call__(self, h, c):
        sh, sc, g = jnp.split(nn.Dense(3 * self.w, kernel_init=nn.initializers.zeros)(nn.silu(c)), 3, -1)
        y = nn.LayerNorm(use_scale=False, use_bias=False)(h) * (1 + sc) + sh
        y = nn.Dense(self.w)(nn.silu(nn.Dense(self.w)(y)))
        return h + g * y


# ---------------------------------------------------------------- Track B data
def load_D2(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def density_to_iv(dens_z, zgrid, xgrid, taus, sig_ref=0.2):
    """dens_z [..., m, G] density in std units per maturity -> IV [..., m, n] at D2 log strikes."""
    from bench.finance import implied_vol
    dz = zgrid[1] - zgrid[0]
    out = np.empty(dens_z.shape[:-1] + (xgrid.shape[1],))
    for j, t in enumerate(taus):
        x = zgrid * sig_ref * math.sqrt(t)
        w = dens_z[..., j, :] * dz
        w = w / w.sum(-1, keepdims=True)
        S = np.exp(x) / (w * np.exp(x)).sum(-1, keepdims=True)  # rescale so the forward is exactly 1
        K = np.exp(xgrid[j])
        put = (np.maximum(K[:, None] - S[..., None, :], 0) * w[..., None, :]).sum(-1)
        call = (np.maximum(S[..., None, :] - K[:, None], 0) * w[..., None, :]).sum(-1)
        p = np.where(xgrid[j] < 0, put, call)
        out[..., j, :] = implied_vol(p, np.broadcast_to(xgrid[j], p.shape), t)
    return out
