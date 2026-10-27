"""Real-data backtest for Track A models (the pitch's "compare against real payoffs and Black-Scholes-Merton").

Input: a real export (bench.real) and a model's samples.npy [E, N, H] for its eval_ctx.npy. No ground-truth
prices exist on real data, so the scores are about forecasts that can be checked against what happened:

  realized payoffs   for strikes K = S0 exp(k * sigma_ctx * sqrt(H)) (sigma_ctx = the context's realized daily vol),
                     compare the model's expected payoff with the payoff that was actually realized 20 days later,
                     averaged over many dates (unbiased model => mean difference ~ 0, within its standard error).
  digital calibration  model P(S_T > K) vs how often S_T > K actually happened.
  distribution       CRPS of the terminal return; PIT coverage of the model's 90% interval (target 0.90);
                     left-tail exceedances at 1% and 5% (targets 0.01 and 0.05); QLIKE vol-forecast loss.
  prices (pitch)     discounted expected payoff exp(-rT) E[payoff], both as-is (real-world paths) and
                     risk-neutral adjusted (paths rescaled so E[S_T] = S0 e^{rT}); model-implied vol vs sigma_ctx.
                     As-is prices are what the pitch calls "discounted EV"; only the adjusted ones are comparable
                     with market prices (bench.market).

  python -m bench.backtest --data bench_data/real/megacap_tech --samples runs/real_a6/samples.npy
  python -m bench.backtest table "results/backtest/*.json"
"""
import os as _os
import sys as _sys

if __package__ in (None, ""):  # also allow `python path/to/file.py`, not only `python -m pkg.file`
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import glob
import json
import math
import os

import numpy as np

from bench.finance import implied_vol


def crps_samples(x, y, n=2000, rng=None):
    """Sample CRPS per row: E|X - y| - 0.5 E|X - X'|. x [E, N], y [E]."""
    rng = rng or np.random.default_rng(0)
    if x.shape[1] > n:
        x = x[:, rng.choice(x.shape[1], n, replace=False)]
    t1 = np.abs(x - y[:, None]).mean(1)
    xs = np.sort(x, 1)
    m = xs.shape[1]
    w = (2 * np.arange(1, m + 1) - m - 1) / (m * m)
    t2 = (xs * w).sum(1)  # = 0.5 E|X - X'| for the empirical distribution
    return t1 - t2


def payoff(S, K, put):
    """OTM payoffs. S: [E, N] paths or [E] realized; K: [E, n]; put: [n] bool -> [E, n, N] or [E, n]."""
    if S.ndim == 2:
        S, K, put = S[:, None, :], K[..., None], put[None, :, None]
    else:
        S, put = S[:, None], put[None, :]
    return np.where(put, np.maximum(K - S, 0), np.maximum(S - K, 0))


def score(data, samples_path, label=None, out="results/backtest"):
    cfg = json.load(open(os.path.join(data, "config.json")))
    assert cfg.get("kind") == "real", "bench.backtest is for real-data exports; use bench.trackA for synthetic"
    ctx = np.load(os.path.join(data, "eval_ctx.npy")).astype(np.float64)
    real = np.load(os.path.join(data, "eval_realized.npy")).astype(np.float64)
    rr = np.load(os.path.join(data, "eval_r.npy"))
    meta = json.load(open(os.path.join(data, "eval_meta.json")))
    smp = np.load(samples_path).astype(np.float64)
    E, N, H = smp.shape
    assert E == ctx.shape[0], f"samples have {E} contexts, data has {ctx.shape[0]}"
    mp = os.path.join(os.path.dirname(samples_path), "meta.json")
    mm = json.load(open(mp)) if os.path.exists(mp) else {}
    label = label or mm.get("label") or os.path.basename(os.path.dirname(samples_path))
    ks = np.array(cfg["ks"])
    T = H / 252
    sig = np.sqrt(np.mean(ctx ** 2, 1))                          # daily realized vol of the context
    G = np.exp(smp.sum(-1))                                      # [E, N] model terminal gross return
    g = np.exp(real.sum(-1))                                     # [E] realized
    K = np.exp(ks[None, :] * sig[:, None] * math.sqrt(H))        # [E, n] strikes (S0 = 1)
    put = ks < 0
    pm = payoff(G, K, put).mean(-1)                              # [E, n] model E[payoff] (real-world)
    pr = payoff(g, K, put)                                       # [E, n] realized payoff
    d = pm - pr
    res = dict(label=label, n_contexts=E, n_paths=N, ks=ks.tolist())
    res["payoff_bias"] = d.mean(0).tolist()                      # in units of S0
    res["payoff_bias_se"] = (d.std(0) / math.sqrt(E)).tolist()
    res["payoff_bias_t"] = (d.mean(0) / np.maximum(d.std(0) / math.sqrt(E), 1e-12)).tolist()
    res["payoff_ratio_model_over_realized"] = (pm.mean(0) / np.maximum(pr.mean(0), 1e-12)).tolist()
    # digital calibration P(S_T > K)
    pdig = (G[:, None, :] > K[..., None]).mean(-1)
    rdig = (g[:, None] > K).astype(float)
    res["digital_model"] = pdig.mean(0).tolist(); res["digital_realized"] = rdig.mean(0).tolist()
    # distribution of the terminal log return
    X, y = np.log(G), np.log(g)
    res["crps_mean"] = float(crps_samples(X, y).mean())
    u = (X <= y[:, None]).mean(1)
    res["pit"] = u.tolist()
    res["coverage90"] = float(np.mean((u > 0.05) & (u < 0.95)))
    res["exceed_1pct"] = float(np.mean(u < 0.01)); res["exceed_5pct"] = float(np.mean(u < 0.05))
    v = np.maximum(X.var(1), 1e-12)
    res["qlike"] = float(np.mean(np.log(v) + y ** 2 / v))
    # prices: discounted EV as-is and risk-neutral adjusted; model-implied vol relative to sigma_ctx
    disc = np.exp(-rr * T)[:, None]
    price_P = disc * pm
    Gq = G * (np.exp(rr * T)[:, None] / G.mean(1, keepdims=True))
    pq = payoff(Gq, K, put).mean(-1)
    price_Q = disc * pq
    F = np.exp(rr * T)[:, None]
    x = np.log(K / F)
    ivq = implied_vol(pq / F, x, T)                              # undiscounted, forward-normalized
    res["price_discounted_P_mean"] = price_P.mean(0).tolist()
    res["price_discounted_Q_mean"] = price_Q.mean(0).tolist()
    res["iv_Q_over_ctx_vol_by_k"] = np.nanmean(ivq / (sig[:, None] * math.sqrt(252)), 0).tolist()
    res["drift_annual_model_mean"] = float(np.mean(np.log(G.mean(1))) / T)
    tick = sorted(set(m["ticker"] for m in meta))
    cr = crps_samples(X, y)
    res["by_ticker"] = {t: dict(crps=float(cr[[m["ticker"] == t for m in meta]].mean()),
                                coverage90=float(np.mean(((u > 0.05) & (u < 0.95))[[m["ticker"] == t for m in meta]])))
                        for t in tick}
    res.update({k: v for k, v in mm.items() if k != "label"})
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, f"{label}.json"), "w"), indent=1)
    print(f"[{label}] CRPS {res['crps_mean']:.5f}  cov90 {res['coverage90']:.2f}  exceed 1%/5% {res['exceed_1pct']:.3f}/{res['exceed_5pct']:.3f}  "
          f"QLIKE {res['qlike']:.3f}\n    payoff bias t-stats by k: " +
          "  ".join(f"{k:+.1f}:{t:+.1f}" for k, t in zip(ks, res["payoff_bias_t"])))
    return res


def table(paths):
    rows = [json.load(open(p)) for g_ in paths for p in glob.glob(g_)]
    cols = ["crps_mean", "coverage90", "exceed_1pct", "exceed_5pct", "qlike"]
    print("| model | " + " | ".join(cols) + " | worst |payoff bias t| |")
    print("|---|" + "---|" * (len(cols) + 1))
    for r in sorted(rows, key=lambda r: r["crps_mean"]):
        print(f"| {r['label']} | " + " | ".join(f"{r[c]:.4g}" for c in cols) + f" | {max(abs(t) for t in r['payoff_bias_t']):.1f} |")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="score", choices=["score", "table"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--data"); ap.add_argument("--samples"); ap.add_argument("--label")
    ap.add_argument("--out", default="results/backtest")
    a = ap.parse_args(argv)
    if a.cmd == "table":
        table(a.paths)
    else:
        score(a.data, a.samples, a.label, a.out)


if __name__ == "__main__":
    main()
