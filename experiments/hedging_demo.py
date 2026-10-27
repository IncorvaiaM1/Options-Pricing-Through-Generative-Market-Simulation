"""Hedging-error demo on the GARCH-t world, no trained model needed.

Compares three pricers on the same true continuation paths:
  oracle      true MC price per context, hedged at its own implied vol (the floor)
  bs_longrun  Black-Scholes at the long-run vol (the v0/v1 baseline)
  bs_state    Black-Scholes at the current GARCH state vol (knows the state, misses the smile)

  python -m experiments.hedging_demo --n_ctx 100 --n_true 2000

To score a trained model, build its [E, n_k] price and IV arrays the way ardiff/evaluate.py
already does (otm prices from generated paths -> implied vol) and pass them to
bench.hedging.hedge_scores with the same true paths.
"""
import os as _os
import sys as _sys

if __package__ in (None, ""):  # also allow `python path/to/file.py`, not only `python -m pkg.file`
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import math

import numpy as np

from ardiff import data as D
from bench import finance as bf
from bench import hedging as bh


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ctx", type=int, default=100)
    ap.add_argument("--n_true", type=int, default=2000, help="true paths per context for hedging")
    ap.add_argument("--n_mc", type=int, default=20000, help="MC paths per context for the true price")
    ap.add_argument("--ctx_len", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    p = D.GarchT()
    rng = np.random.default_rng(a.seed)
    _, next_var = D.simulate(p, a.n_ctx, a.ctx_len, rng)
    H, tau = a.horizon, a.horizon / 252
    sd_H = p.annual_vol * math.sqrt(tau)
    ks = np.arange(-4, 5)
    x = ks * sd_H

    # true prices from the exact state
    X_T = D.true_terminal_logret(p, next_var, H, a.n_mc, rng)            # [E, n_mc]
    true_px = bf.otm_from_logret(X_T, np.broadcast_to(x, (a.n_ctx, len(x))))  # [E, n_k]
    true_iv = bh.iv_from_prices(true_px, x[None, :], tau)

    lr_iv = np.full_like(true_px, p.annual_vol)
    lr_px = bf.bs_otm(lr_iv, x[None, :], tau)
    st_iv = np.repeat(np.sqrt(next_var * 252)[:, None], len(x), axis=1)
    st_px = bf.bs_otm(st_iv, x[None, :], tau)

    paths = bh.true_continuation_paths(p, next_var, H, a.n_true, rng)  # [E, n_true, H]
    rows = {
        "oracle": bh.hedge_scores(paths, x, tau, true_px, true_iv),
        "bs_longrun": bh.hedge_scores(paths, x, tau, lr_px, lr_iv),
        "bs_state": bh.hedge_scores(paths, x, tau, st_px, st_iv),
    }

    # The GARCH-t spot is not a martingale (log returns are mean zero, so E[S_T] > S0 by
    # roughly half the summed variance, ~14 bps over 20 days), so even the oracle's mean
    # hedged P&L is off zero by the drift the delta hedge picks up. Comparing against the
    # oracle on the same paths cancels that out.
    o = rows["oracle"]
    hdr = "| model | " + " | ".join(f"k={k}" for k in ks) + " |"
    sep = "|---" * (len(ks) + 1) + "|"
    tables = (
        ("Mean hedged P&L minus oracle (bps of S0; + = priced rich)", lambda r: r["mean"] - o["mean"]),
        ("Std of hedged P&L / oracle std (1.00 = hedges as well as the truth)", lambda r: r["std"] / o["std"]),
        ("Raw mean hedged P&L (bps of S0; oracle row = drift + MC noise)", lambda r: r["mean"]),
        ("Raw std of hedged P&L (bps of S0)", lambda r: r["std"]),
    )
    for title, f in tables:
        print(f"\n## {title}\n\n{hdr}\n{sep}")
        for name, r in rows.items():
            print(f"| {name} | " + " | ".join(f"{v:.2f}" for v in f(r)) + " |")


if __name__ == "__main__":
    main()
