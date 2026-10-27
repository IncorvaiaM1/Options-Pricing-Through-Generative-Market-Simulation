"""A0: classical baselines that write samples.npy, so they are scored exactly like the generative models.

  bsm        : Black-Scholes-Merton. Lognormal paths at the context's realized vol (last 64 days), martingale drift.
  bootstrap  : resample the context's own daily returns i.i.d. (historical simulation).
  garch      : GARCH(1,1) with Student-t shocks, parameters fitted by maximum likelihood on the training windows
               (pooled), state filtered through each context, then simulated forward. On the synthetic data this is
               the correct model family, so it is close to an oracle there; on real data it is the standard
               econometric benchmark.

Works on both bench/A (synthetic) and real-data exports (bench.real), since both share the file format.

  python -m starters.a0_baselines --method bsm --data bench_data/A --out runs/bsm
"""
import os as _os
import sys as _sys

if __package__ in (None, ""):  # also allow `python path/to/file.py`, not only `python -m pkg.file`
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


def realized_vol(ctx):
    return np.sqrt(np.mean(ctx ** 2, axis=1))  # daily


def gen_bsm(ctx, N, H, rng):
    s = realized_vol(ctx)[:, None, None]
    z = rng.standard_normal((ctx.shape[0], N, H))
    return s * z - 0.5 * s ** 2


def gen_bootstrap(ctx, N, H, rng):
    E, L = ctx.shape
    idx = rng.integers(0, L, (E, N, H))
    return np.take_along_axis(ctx[:, None, :].repeat(N, 1), idx, axis=2)


def garch_nll(theta, X):
    """X: [n, T] returns. theta = (log omega, logit-ish alpha, beta, log(nu-2)). Pooled Student-t GARCH(1,1)."""
    w = np.exp(theta[0]); a = 1 / (1 + np.exp(-theta[1])) * 0.3; b = 1 / (1 + np.exp(-theta[2])) * (0.999 - a)
    nu = 2.05 + np.exp(theta[3])
    var = np.full(X.shape[0], X.var())
    ll = 0.0
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    for t in range(X.shape[1]):
        r = X[:, t]
        ll += np.sum(c - 0.5 * np.log(var) - (nu + 1) / 2 * np.log1p(r ** 2 / (var * (nu - 2))))
        var = w + a * r ** 2 + b * var
    return -ll / X.size


def fit_garch(train, rng, n_fit=2000):
    X = train[rng.choice(train.shape[0], min(n_fit, train.shape[0]), replace=False)].astype(np.float64)
    v = X.var()
    x0 = np.array([np.log(v * 0.05), 0.0, 2.0, np.log(4.0)])
    r = minimize(garch_nll, x0, args=(X,), method="Nelder-Mead", options=dict(maxiter=600, xatol=1e-4, fatol=1e-7))
    th = r.x
    w = np.exp(th[0]); a = 1 / (1 + np.exp(-th[1])) * 0.3; b = 1 / (1 + np.exp(-th[2])) * (0.999 - a)
    nu = 2.05 + np.exp(th[3])
    return dict(omega=float(w), alpha=float(a), beta=float(b), nu=float(nu), nll=float(r.fun))


def gen_garch(ctx, N, H, rng, p):
    E = ctx.shape[0]
    var = np.full(E, ctx.var(axis=1).mean())
    for t in range(ctx.shape[1]):
        var = p["omega"] + p["alpha"] * ctx[:, t] ** 2 + p["beta"] * var
    var = np.repeat(var[:, None], N, 1)
    out = np.empty((E, N, H))
    s = np.sqrt(p["nu"] / (p["nu"] - 2))
    for t in range(H):
        r = np.sqrt(var) * rng.standard_t(p["nu"], (E, N)) / s
        out[:, :, t] = r
        var = p["omega"] + p["alpha"] * r ** 2 + p["beta"] * var
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["bsm", "bootstrap", "garch"])
    ap.add_argument("--data", default="bench_data/A")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_paths", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = args.out or f"runs/{args.method}"
    cfg = json.load(open(os.path.join(args.data, "config.json")))
    ctx = np.load(os.path.join(args.data, "eval_ctx.npy")).astype(np.float64)
    H = cfg["horizon"]
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    meta = dict(label=f"baseline_{args.method}", n_params=0)
    if args.method == "bsm":
        fut = gen_bsm(ctx, args.n_paths, H, rng)
    elif args.method == "bootstrap":
        fut = gen_bootstrap(ctx, args.n_paths, H, rng)
    else:
        p = fit_garch(np.load(os.path.join(args.data, "train.npy")), rng)
        print("fitted GARCH-t:", {k: round(v, 6) for k, v in p.items()}, flush=True)
        meta["garch_fit"] = p
        meta["n_params"] = 4
        fut = gen_garch(ctx, args.n_paths, H, rng, p)
    meta["train_seconds"] = 0.0
    meta["gen_seconds"] = time.time() - t0
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "samples.npy"), fut.astype(np.float32))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"wrote {out}/samples.npy {fut.shape}")


if __name__ == "__main__":
    main()
