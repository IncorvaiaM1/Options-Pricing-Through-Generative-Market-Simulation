"""Track A harness: framework-agnostic scoring of market simulators.

1) export the shared data once:
     python -m bench.trackA export --out bench_data/A
   writes train.npy [n_train, L+H] and val.npy (daily log returns), eval_ctx.npy [E, L], config.json,
   truth.npz (exact-in-distribution prices and path statistics from the known GARCH state).
   Same data as ardiff/train.py with the same defaults, so A1/A2 checkpoints line up.

2) any model (PyTorch, JAX, anything) writes samples.npy [E, N, H] of FUTURE daily log returns
   (raw units, not normalized) for the E rows of eval_ctx.npy, plus optional meta.json
   {"label": ..., "n_params": ..., "train_seconds": ..., "gen_seconds": ...}

3) score:
     python -m bench.trackA score --data bench_data/A --samples runs/a6/samples.npy --label a6_pure_diffusion
     python -m bench.trackA table results/trackA/*.json
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
import warnings

import numpy as np

from ardiff import data as D
from ardiff.evaluate import implied_vol, otm_prices

KS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


def path_stats(paths):
    """paths: [E, N, H] daily log returns -> dict of per-context statistics."""
    lr = paths.sum(-1)
    S = np.exp(np.cumsum(paths, -1))
    asian = np.maximum(S.mean(-1) - 1, 0).mean(-1)
    upout = (np.maximum(S[..., -1] - 1, 0) * (S.max(-1) < 1.10)).mean(-1)
    a = np.abs(paths)
    a = a - a.mean((-1, -2), keepdims=True)  # demean per context over paths and time (per-path demeaning biases short windows)
    acf1 = (a[..., 1:] * a[..., :-1]).mean((-1, -2)) / np.maximum((a ** 2).mean((-1, -2)), 1e-30)
    return dict(terminal_q=np.quantile(lr, [0.001, 0.01, 0.99, 0.999], axis=-1).T,  # [E, 4]
                asian_atm=asian, upout_110=upout, acf_abs_lag1=acf1)


def export(args):
    os.makedirs(args.out, exist_ok=True)
    g = D.GarchT(nu=args.nu)
    sp = D.make_splits(g, args.ctx_len, args.horizon, args.n_train, args.n_val, args.n_eval, args.data_seed)
    np.save(os.path.join(args.out, "train.npy"), sp["train"].astype(np.float32))
    np.save(os.path.join(args.out, "val.npy"), sp["val"].astype(np.float32))
    np.save(os.path.join(args.out, "eval_ctx.npy"), sp["eval_ctx"].astype(np.float32))
    np.save(os.path.join(args.out, "eval_next_var.npy"), sp["eval_next_var"])
    cfg = dict(kind="synthetic", r=0.0, ctx_len=args.ctx_len, horizon=args.horizon, n_train=args.n_train, n_val=args.n_val,
               n_eval=args.n_eval, nu=args.nu, data_seed=args.data_seed, scale=sp["scale"], garch=g.to_dict(), ks=KS)
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)

    # truth: simulate full continuation paths from the known state, in chunks
    rng = np.random.default_rng(args.truth_seed)
    H, E = args.horizon, args.n_eval
    sd_H = g.annual_vol * math.sqrt(H / 252)
    P = np.zeros((E, len(KS))); stats = []
    n_keep = min(args.n_hedge_paths, args.n_mc)
    keep_paths = np.empty((E, n_keep, H), np.float32)  # true continuation paths per context: plots + hedging metric
    for e in range(E):
        var = np.full(args.n_mc, sp["eval_next_var"][e])
        paths = np.empty((args.n_mc, H))
        for t in range(H):
            r = np.sqrt(var) * D._innov(rng, g.nu, args.n_mc)
            paths[:, t] = r
            var = g.omega + g.alpha * r ** 2 + g.beta * var
        P[e] = otm_prices(paths.sum(-1)[None], np.array(KS), sd_H)[0][0]
        stats.append(path_stats(paths[None]))
        keep_paths[e] = paths[:n_keep]
    st = {k: np.concatenate([s[k] for s in stats]) for k in stats[0]}
    iv = implied_vol(P, np.array(KS), sd_H, H / 252)
    np.savez(os.path.join(args.out, "truth.npz"), price=P, iv=iv, state_vol=np.sqrt(sp["eval_next_var"] * 252),
             truth_paths=keep_paths, **st)
    print(f"exported to {args.out}: train {sp['train'].shape}, eval {E} contexts, truth from {args.n_mc:,} paths each")


def score(args):
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN strikes for weak models are expected
    cfg = json.load(open(os.path.join(args.data, "config.json")))
    tr = dict(np.load(os.path.join(args.data, "truth.npz")))
    smp = np.load(args.samples).astype(np.float64)
    E, N, H = smp.shape
    assert E == cfg["n_eval"] and H == cfg["horizon"], f"samples shape {smp.shape} does not match eval set"
    meta = {}
    mp = os.path.join(os.path.dirname(args.samples), "meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    label = args.label or meta.pop("label", None) or os.path.basename(os.path.dirname(args.samples))
    meta.pop("label", None)
    ks = np.array(cfg["ks"])
    sd_H = cfg["garch"]["annual_vol"] * math.sqrt(H / 252)
    P, _ = otm_prices(smp.sum(-1), ks, sd_H)
    iv = implied_vol(P, ks, sd_H, H / 252)
    st = path_stats(smp)
    ive = np.abs(iv - tr["iv"]) * 100
    stressed = tr["state_vol"] >= np.quantile(tr["state_vol"], 2 / 3)
    wing = np.abs(ks) >= 2
    step_abs = np.abs(smp) / cfg["scale"]
    res = dict(
        label=label, n_paths=N, ks=ks.tolist(),
        mean_abs_iv_err_by_k=np.nanmean(ive, 0).tolist(),
        mean_abs_iv_err_by_k_stressed=np.nanmean(ive[stressed], 0).tolist(),
        iv_err_atm=float(np.nanmean(ive[:, np.abs(ks) <= 1])), iv_err_wings=float(np.nanmean(ive[:, wing])),
        iv_err_wings_stressed=float(np.nanmean(ive[stressed][:, wing])),
        iv_nan_frac=float(np.isnan(iv).mean()),
        terminal_q_abs_err=np.mean(np.abs(st["terminal_q"] - tr["terminal_q"]), 0).tolist(),
        asian_rel_err=float(np.mean(np.abs(st["asian_atm"] - tr["asian_atm"]) / tr["asian_atm"])),
        upout_rel_err=float(np.mean(np.abs(st["upout_110"] - tr["upout_110"]) / np.maximum(tr["upout_110"], 1e-12))),
        acf_abs_lag1_model=float(st["acf_abs_lag1"].mean()), acf_abs_lag1_true=float(tr["acf_abs_lag1"].mean()),
        explosive_rate=float((step_abs > 50).mean()), step_max_abs_std=float(step_abs.max()), **meta)
    if not getattr(args, "no_hedge", False):
        res.update(hedging_block(args.data, tr, ks, sd_H, H, P / 100.0, iv, stressed))
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, f"{label}.json"), "w"), indent=1)
    if "hedge_std_ratio_by_k" in res:
        print(f"[{label}] hedging vs oracle: std ratio ATM {res['hedge_std_ratio_atm']:.3f}  wings {res['hedge_std_ratio_wings']:.3f}; "
              f"mean P&L - oracle (bps) by k: " + " ".join(f"{v:+.1f}" for v in res["hedge_mean_minus_oracle_bps_by_k"]))
    print(f"[{label}] IV err ATM {res['iv_err_atm']:.2f}  wings {res['iv_err_wings']:.2f}  "
          f"wings|stressed {res['iv_err_wings_stressed']:.2f} vol pts; asian rel err {res['asian_rel_err']:.1%}; "
          f"ACF|r| {res['acf_abs_lag1_model']:.3f} (true {res['acf_abs_lag1_true']:.3f}); explosive {res['explosive_rate']:.2e}")
    return res


def hedging_block(data, tr, ks, sd_H, H, model_px, model_iv, stressed):
    """Delta-hedging error on the TRUE continuation paths, reported against the oracle (true price + true IV hedged
    on the same paths). Uses fatnarwhal-quant (the FatNarwhal tools' Black-Scholes); skipped if it isn't installed.
    Against-the-oracle is required: the synthetic spot is not a martingale (mean-zero log returns drift up by about
    half the summed variance), and the delta hedge picks that drift up for every pricer alike."""
    try:
        from bench import hedging as bh
    except ImportError:
        print("fatnarwhal-quant not installed: skipping the hedging metric (pip install fatnarwhal-quant)")
        return {}
    if "truth_paths" not in tr or tr["truth_paths"].shape[1] < 200:
        print("truth.npz has too few true paths for hedging; re-export with a newer bench.trackA")
        return {}
    paths = tr["truth_paths"].astype(np.float64)
    x = ks * sd_H
    tau = H / 252
    cache = os.path.join(data, "hedge_oracle.npz")
    if os.path.exists(cache):
        o = dict(np.load(cache))
    else:
        oa = bh.hedge_scores(paths, x, tau, tr["price"] / 100.0, tr["iv"])
        os_ = bh.hedge_scores(paths[stressed], x, tau, tr["price"][stressed] / 100.0, tr["iv"][stressed])
        o = {f"all_{k}": v for k, v in oa.items()} | {f"st_{k}": v for k, v in os_.items()}
        np.savez(cache, **o)
    a = bh.hedge_scores(paths, x, tau, model_px, model_iv)
    st = bh.hedge_scores(paths[stressed], x, tau, model_px[stressed], model_iv[stressed])
    wing = np.abs(ks) >= 2
    ratio = a["std"] / o["all_std"]
    return dict(hedge_mean_minus_oracle_bps_by_k=(a["mean"] - o["all_mean"]).tolist(),
                hedge_std_ratio_by_k=ratio.tolist(),
                hedge_mean_minus_oracle_bps_by_k_stressed=(st["mean"] - o["st_mean"]).tolist(),
                hedge_std_ratio_by_k_stressed=(st["std"] / o["st_std"]).tolist(),
                hedge_std_ratio_atm=float(np.nanmean(ratio[np.abs(ks) <= 1])),
                hedge_std_ratio_wings=float(np.nanmean(ratio[wing])),
                hedge_abs_mean_minus_oracle_bps_wings=float(np.nanmean(np.abs(a["mean"] - o["all_mean"])[wing])),
                hedge_oracle_std_bps_by_k=o["all_std"].tolist())


def table(paths):
    rows = [json.load(open(p)) for g in paths for p in glob.glob(g)]
    cols = ["iv_err_atm", "iv_err_wings", "iv_err_wings_stressed", "hedge_std_ratio_wings", "hedge_abs_mean_minus_oracle_bps_wings",
            "asian_rel_err", "acf_abs_lag1_model", "explosive_rate", "gen_seconds"]
    print("| model | " + " | ".join(cols) + " |"); print("|---|" + "---|" * len(cols))
    for r in sorted(rows, key=lambda r: r["iv_err_wings"]):
        print(f"| {r['label']} | " + " | ".join("" if r.get(c) is None else f"{r[c]:.3g}" for c in cols) + " |")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", default="bench_data/A")
    e.add_argument("--ctx_len", type=int, default=64); e.add_argument("--horizon", type=int, default=20)
    e.add_argument("--n_train", type=int, default=20000); e.add_argument("--n_val", type=int, default=2000)
    e.add_argument("--n_eval", type=int, default=32); e.add_argument("--nu", type=float, default=5.0)
    e.add_argument("--data_seed", type=int, default=0); e.add_argument("--truth_seed", type=int, default=123)
    e.add_argument("--n_mc", type=int, default=200000)
    e.add_argument("--n_hedge_paths", type=int, default=2000, help="true paths kept per context for hedging and plots")
    s = sub.add_parser("score")
    s.add_argument("--data", default="bench_data/A"); s.add_argument("--samples", required=True)
    s.add_argument("--label", default=None); s.add_argument("--out", default="results/trackA")
    s.add_argument("--no_hedge", action="store_true", help="skip the delta-hedging metric")
    t = sub.add_parser("table"); t.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)
    {"export": export, "score": score, "table": lambda a: table(a.paths)}[args.cmd](args)


if __name__ == "__main__":
    main()
