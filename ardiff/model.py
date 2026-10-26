"""Causal transformer backbone with two interchangeable output heads.

  head="diffusion": continuous next-step distribution via a small conditional DDPM
                    (MAR-style, Li et al. 2024), no quantization anywhere.
  head="token":     categorical over return bins (Kronos/Chronos-style discretization).

Both heads share the exact same backbone and continuous input embedding, so any pricing
difference comes from the output parameterization, which is the thesis.
"""
import math

import jax
import jax.numpy as jnp
import flax.linen as nn


class Block(nn.Module):
    d: int
    n_heads: int
    mlp_mult: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(self, h, mask, train):
        a = nn.LayerNorm()(h)
        a = nn.MultiHeadDotProductAttention(num_heads=self.n_heads, dropout_rate=self.dropout,
                                            deterministic=not train)(a, a, mask=mask)
        h = h + a
        m = nn.LayerNorm()(h)
        m = nn.Dense(self.mlp_mult * self.d)(m)
        m = nn.gelu(m)
        m = nn.Dense(self.d)(m)
        m = nn.Dropout(self.dropout, deterministic=not train)(m)
        return h + m


class Backbone(nn.Module):
    """x: [B, T] normalized returns -> z: [B, T, d]; z[:, t] summarizes x[:, :t+1] and predicts x[:, t+1]."""
    d: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_len: int = 256
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, train=False):
        B, T = x.shape
        # continuous input embedding; sign-preserving log feature helps with fat tails
        feats = jnp.stack([x, jnp.sign(x) * jnp.log1p(jnp.abs(x))], axis=-1)
        h = nn.Dense(self.d)(feats)
        pos = self.param("pos", nn.initializers.normal(0.02), (self.max_len, self.d))
        h = h + pos[None, :T]
        mask = nn.make_causal_mask(jnp.ones((B, T)))
        for _ in range(self.n_layers):
            h = Block(self.d, self.n_heads, dropout=self.dropout)(h, mask, train)
        return nn.LayerNorm()(h)


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(half) / half)
    args = t.astype(jnp.float32)[..., None] * freqs
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


class ResBlockAdaLN(nn.Module):
    w: int

    @nn.compact
    def __call__(self, h, c):
        shift, scale, gate = jnp.split(nn.Dense(3 * self.w, kernel_init=nn.initializers.zeros)(nn.silu(c)), 3, -1)
        y = nn.LayerNorm(use_scale=False, use_bias=False)(h) * (1 + scale) + shift
        y = nn.Dense(self.w)(y)
        y = nn.silu(y)
        y = nn.Dense(self.w)(y)
        return h + gate * y


class DiffusionHead(nn.Module):
    """eps_theta(x_t, t, z): predicts the noise added to a scalar next-step return."""
    width: int = 256
    depth: int = 3

    @nn.compact
    def __call__(self, x_t, t, z):
        c = nn.Dense(self.width)(z) + nn.Dense(self.width)(nn.silu(nn.Dense(self.width)(timestep_embedding(t, self.width))))
        h = nn.Dense(self.width)(x_t[..., None])
        for _ in range(self.depth):
            h = ResBlockAdaLN(self.width)(h, c)
        h = nn.LayerNorm(use_scale=False, use_bias=False)(h)
        shift, scale = jnp.split(nn.Dense(2 * self.width, kernel_init=nn.initializers.zeros)(nn.silu(c)), 2, -1)
        h = h * (1 + scale) + shift
        return nn.Dense(1, kernel_init=nn.initializers.zeros)(h)[..., 0]


class TokenHead(nn.Module):
    n_bins: int = 256

    @nn.compact
    def __call__(self, z):
        return nn.Dense(self.n_bins)(z)


class ARModel(nn.Module):
    head: str = "diffusion"          # "diffusion" | "token"
    d: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_len: int = 256
    dropout: float = 0.0
    head_width: int = 256
    head_depth: int = 3
    n_bins: int = 256

    def setup(self):
        self.backbone = Backbone(self.d, self.n_layers, self.n_heads, self.max_len, self.dropout)
        if self.head == "diffusion":
            self.dhead = DiffusionHead(self.head_width, self.head_depth)
        else:
            self.thead = TokenHead(self.n_bins)

    def encode(self, x, train=False):
        return self.backbone(x, train=train)

    def eps(self, x_t, t, z):
        return self.dhead(x_t, t, z)

    def logits(self, z):
        return self.thead(z)

    def __call__(self, x, train=False):  # used for init only
        z = self.encode(x, train)
        if self.head == "diffusion":
            return self.eps(jnp.zeros(z.shape[:-1]), jnp.zeros(z.shape[:-1], jnp.int32), z)
        return self.logits(z)
