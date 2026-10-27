"""Hedging error: score a pricer by how well its price and its vol hedge on TRUE paths.

For each eval context and strike:
  1. sell the option for the model's price,
  2. delta-hedge daily with Black-Scholes deltas at the model's implied vol (sticky strike),
  3. pay the payoff at expiry,
all on continuation paths simulated from the true GARCH state for that context.

  mean P&L ~ model price - true price  (pricing error, signed)
  std  P&L = hedge quality             (lower is better)

The "oracle" row (true price + true IV) is the floor set by daily rebalancing alone.
Black-Scholes math comes from fatnarwhal-quant, the library behind FatNarwhal's tools,
so the delta here is the same delta the site's Black-Scholes tool shows.

Units: r = 0, S0 = 1, P&L reported in basis points of S0.
"""
from __future__ import annotations

import numpy as np
from fatnarwhal_quant import hedging as fnh
from fatnarwhal_quant import normalized as fnn


def true_continuation_paths(p, next_var, horizon, n_paths, rng):
    """GARCH-t continuation returns from each context's known state. [E, n_paths, H]."""
    from ardiff.data import _innov  # noqa: PLC0415  (keeps bench importable without ardiff)

    var = np.repeat(np.asarray(next_var, float)[:, None], n_paths, axis=1)
    out = np.empty(var.shape + (horizon,))
    for j in range(horizon):
        r = np.sqrt(var) * _innov(rng, p.nu, var.shape)
        out[..., j] = r
        var = p.omega + p.alpha * r ** 2 + p.beta * var
    return out


def hedge_scores(true_returns, x, tau, model_price, model_iv):
    """Hedging P&L summary for one pricer.

    true_returns  [E, N, H] true continuation log returns per context
    x             [n_k] log strikes (OTM convention: put for x < 0, call for x >= 0)
    tau           years to expiry (H / 252)
    model_price   [E, n_k] pricer's OTM price per context and strike
    model_iv      [E, n_k] pricer's implied vol per context and strike (NaN -> strike skipped)

    Returns dict of [n_k] arrays in bps of S0: mean, std, rmse (averaged over contexts).
    """
    E = true_returns.shape[0]
    K = np.exp(np.asarray(x, float))
    acc = {k: [] for k in ("mean", "std", "rmse")}
    for e in range(E):
        S = fnh.paths_from_returns(true_returns[e])
        iv = model_iv[e]
        ok = np.isfinite(iv) & np.isfinite(model_price[e])
        row = {k: np.full(K.shape, np.nan) for k in acc}
        if ok.any():
            pnl = fnh.delta_hedge_pnl(S, K[ok], tau, iv[ok], premium=model_price[e][ok], is_call=x[ok] >= 0)
            s = fnh.summarize(pnl)
            for k in acc:
                row[k][ok] = s[k]
        for k in acc:
            acc[k].append(row[k])
    return {k: 1e4 * np.nanmean(np.stack(v), axis=0) for k, v in acc.items()}


def iv_from_prices(price, x, tau):
    """Vectorized OTM implied vol (fatnarwhal-quant), same convention as bench.finance.implied_vol."""
    return fnn.otm_implied_vol(price, x, tau)
