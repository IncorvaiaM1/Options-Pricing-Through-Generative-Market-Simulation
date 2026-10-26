"""Synthetic market with heavy tails, volatility clustering and an exact ground truth.

Daily log returns follow GARCH(1,1) with standardized Student-t innovations:
    r_t        = sigma_t * eps_t,           eps_t ~ t_nu / sqrt(nu / (nu - 2))
    sigma_{t+1}^2 = omega + alpha * r_t^2 + beta * sigma_t^2

Why this process:
  * fat tails (nu ~ 4) are exactly where a tokenized output head has to round coarsely,
  * vol clustering gives an AR model something real to condition on,
  * the conditional variance after a context window is a known state, so the TRUE
    option price for that context can be computed by Monte Carlo from that state.

Prices use S_T = S0 * exp(sum of future log returns), no discounting (r = 0).
"""
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class GarchT:
    annual_vol: float = 0.20
    alpha: float = 0.08
    beta: float = 0.90
    nu: float = 5.0

    @property
    def var_lr(self):  # long-run daily variance
        return self.annual_vol ** 2 / 252

    @property
    def omega(self):
        return self.var_lr * (1 - self.alpha - self.beta)

    def to_dict(self):
        return asdict(self)


def _innov(rng, nu, size):
    return rng.standard_t(nu, size=size) / np.sqrt(nu / (nu - 2))


def simulate(p: GarchT, n_paths, length, rng, burn_in=200, var0=None):
    """Return (returns [n, length], next_var [n]) where next_var is sigma^2 for the step after the window."""
    var = np.full(n_paths, p.var_lr) if var0 is None else np.asarray(var0, dtype=np.float64).copy()
    total = length + (burn_in if var0 is None else 0)
    out = np.empty((n_paths, length))
    for t in range(total):
        r = np.sqrt(var) * _innov(rng, p.nu, n_paths)
        var = p.omega + p.alpha * r ** 2 + p.beta * var
        j = t - (total - length)
        if j >= 0:
            out[:, j] = r
    return out, var


def true_terminal_logret(p: GarchT, next_var, horizon, n_mc, rng):
    """Exact-in-distribution MC of sum of the next `horizon` returns, given the GARCH state. Shape [n_ctx, n_mc]."""
    var = np.repeat(np.asarray(next_var)[:, None], n_mc, axis=1)
    total = np.zeros_like(var)
    for _ in range(horizon):
        r = np.sqrt(var) * _innov(rng, p.nu, var.shape)
        total += r
        var = p.omega + p.alpha * r ** 2 + p.beta * var
    return total


def make_splits(p: GarchT, ctx_len, horizon, n_train, n_val, n_eval, seed=0):
    rng = np.random.default_rng(seed)
    L = ctx_len + horizon
    train, _ = simulate(p, n_train, L, rng)
    val, _ = simulate(p, n_val, L, rng)
    eval_ctx, eval_next_var = simulate(p, n_eval, ctx_len, rng)
    scale = float(train.std())
    return dict(train=train, val=val, eval_ctx=eval_ctx, eval_next_var=eval_next_var, scale=scale)
