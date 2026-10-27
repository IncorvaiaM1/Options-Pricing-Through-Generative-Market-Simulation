"""Track B scorer. Every Track B starter writes a prediction file; this scores it against D2 truth.

Prediction file (npz), all arrays over the TEST split in test.npz order:
  method, scenario   strings
  iv                 [S, m, n] predicted implied vol on the D2 grid            (required)
  iv_lo, iv_hi       [S, m, n] 90% band                                        (optional, gives coverage)
  density            [S, m, G] predicted density on ZGRID (std units)          (optional, GARCH surfaces scored)
  exotics            [S, 4] prices of bench.finance.EXOTICS                    (optional)
  exotics_lo/_hi     [S, 4] band                                               (optional)
  surfaces           [k] indices into the test set, if only a subset was predicted (optional)

  python -m bench.trackB score --test bench_data/D2/test.npz --pred results/trackB/preds/ssvi_sparse10.npz
  python -m bench.trackB table results/trackB/*.json
"""
import os as _os
import sys as _sys

if __package__ in (None, ""):  # also allow `python path/to/file.py`, not only `python -m pkg.file`
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import glob
import json
import os

import numpy as np

from bench.finance import EXOTICS, bs_call
from bench.surfaces import load


def butterfly_violation_rate(iv, xgrid, taus, tol=1e-7):
    """Fraction of interior strike triples where the call price is not convex in K."""
    v, tot = 0, 0
    for s in range(iv.shape[0]):
        for j, t in enumerate(taus):
            sig = iv[s, j]
            ok = np.isfinite(sig)
            if ok.sum() < 3:
                continue
            K = np.exp(xgrid[j][ok])
            c = bs_call(sig[ok], xgrid[j][ok], t)
            slope = np.diff(c) / np.diff(K)
            v += int((np.diff(slope) < -tol).sum())
            tot += len(slope) - 1
    return v / max(tot, 1)


def w1(pdf_a, pdf_b, dz):
    Fa = np.cumsum(pdf_a, -1) * dz
    Fb = np.cumsum(pdf_b, -1) * dz
    return np.abs(Fa - Fb).sum(-1) * dz


def score(test, pred):
    sc = str(pred["scenario"])
    idx = pred["surfaces"] if "surfaces" in pred else np.arange(test["iv"].shape[0])
    true = test["iv"][idx]
    fam = test["family"][idx]
    ks = test["ks"]
    mask = test[f"mask_{sc}"][idx] if f"mask_{sc}" in test else np.zeros_like(true, bool)
    held = ~mask & np.isfinite(true)
    if sc == "full":
        held = np.isfinite(true)  # nothing held out: this is a calibration-fit score
    wing = np.broadcast_to(np.abs(ks)[None, None, :] >= 2, true.shape)
    err = (pred["iv"] - true) * 100
    out = dict(method=str(pred["method"]), scenario=sc, n_surfaces=int(len(idx)))

    def rmse(sel):
        e = err[sel]
        e = e[np.isfinite(e)]
        return float(np.sqrt(np.mean(e ** 2))) if e.size else None

    out["iv_rmse_heldout"] = rmse(held)
    out["iv_rmse_heldout_wings"] = rmse(held & wing)
    for f in ("garch", "heston"):
        out[f"iv_rmse_heldout_{f}"] = rmse(held & (fam == f)[:, None, None])
    out["pred_nan_frac"] = float(np.mean(~np.isfinite(pred["iv"][held])))
    if "iv_lo" in pred:
        lo, hi = pred["iv_lo"], pred["iv_hi"]
        inside = (true >= lo) & (true <= hi)
        out["coverage90_heldout"] = float(inside[held].mean())
        out["band_width_vp"] = float(np.nanmean((hi - lo)[held]) * 100)
    out["butterfly_violation_rate"] = butterfly_violation_rate(pred["iv"], test["xgrid"], test["taus"])
    if "density" in pred:
        g = fam == "garch"
        dz = test["zgrid"][1] - test["zgrid"][0]
        d = w1(pred["density"][g], test["density"][idx][g], dz)
        out["density_w1_std_units"] = float(np.nanmean(d))
        out["density_w1_by_maturity"] = np.nanmean(d, 0).tolist()
    out.update(hedging_scores_B(test, idx, pred["iv"]))
    if "exotics" in pred:
        te = test["exotics"][idx]
        pe = pred["exotics"]
        out["exotic_names"] = list(EXOTICS)
        out["exotic_rel_err_by_family"] = {}
        for f in ("garch", "heston"):
            s = fam == f
            if s.any():
                out["exotic_rel_err_by_family"][f] = (np.nanmean(np.abs(pe[s] - te[s]) / te[s], 0)).tolist()
        if "exotics_lo" in pred:
            out["exotic_coverage"] = np.nanmean((te >= pred["exotics_lo"]) & (te <= pred["exotics_hi"]), 0).tolist()
    return out


def hedging_scores_B(test, idx, pred_iv, maturity_days=20):
    """Delta-hedge each grid strike of the 20-day smile at the model's IV, sold at the model's price, on the TRUE
    paths of each test surface; report against the oracle (true IV) on the same paths. Needs fatnarwhal-quant."""
    try:
        from bench import hedging as bh
        from bench.finance import bs_otm
    except ImportError:
        return {}
    if "paths_lnS" not in test:
        return {}
    j = list(test["taus_days"]).index(maturity_days)
    x, tau = test["xgrid"][j], test["taus"][j]
    paths = np.diff(test["paths_lnS"][idx][:, :, :maturity_days + 1].astype(np.float64), axis=-1)  # [S, N, d] returns
    t_iv = test["iv"][idx][:, j]
    m_iv = np.where(np.isfinite(pred_iv[:, j]), pred_iv[:, j], np.nan)
    o = bh.hedge_scores(paths, x, tau, bs_otm(t_iv, x[None], tau), t_iv)
    a = bh.hedge_scores(paths, x, tau, bs_otm(m_iv, x[None], tau), m_iv)
    wing = np.abs(test["ks"]) >= 2
    r = a["std"] / o["std"]
    return dict(hedge20_std_ratio_by_k=r.tolist(), hedge20_mean_minus_oracle_bps_by_k=(a["mean"] - o["mean"]).tolist(),
                hedge20_std_ratio_wings=float(np.nanmean(r[wing])), hedge20_std_ratio_atm=float(np.nanmean(r[np.abs(test["ks"]) <= 1])),
                hedge20_abs_mean_minus_oracle_bps=float(np.nanmean(np.abs(a["mean"] - o["mean"]))))


def save_pred(path, method, scenario, iv, **extra):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, method=method, scenario=scenario, iv=iv, **extra)


def score_and_save(test_path, pred_path, out_dir="results/trackB"):
    test = load(test_path)
    pred = load(pred_path)
    res = score(test, pred)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{res['method']}_{res['scenario']}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    keys = [k for k in ("iv_rmse_heldout", "iv_rmse_heldout_wings", "coverage90_heldout", "density_w1_std_units",
                        "butterfly_violation_rate", "hedge20_std_ratio_wings", "hedge20_abs_mean_minus_oracle_bps")
            if res.get(k) is not None]
    print(f"[{res['method']} | {res['scenario']}] " + "  ".join(f"{k}={res[k]:.4g}" for k in keys))
    if "exotic_rel_err_by_family" in res:
        for f, v in res["exotic_rel_err_by_family"].items():
            print(f"    exotic rel err ({f}): " + ", ".join(f"{n}={e:.1%}" for n, e in zip(EXOTICS, v)))
    return res


def table(paths):
    rows = [r for r in (json.load(open(p)) for p in sorted(paths)) if "scenario" in r and "method" in r]  # skip e.g. b4 summaries
    cols = ["iv_rmse_heldout", "iv_rmse_heldout_wings", "iv_rmse_heldout_garch", "iv_rmse_heldout_heston",
            "coverage90_heldout", "density_w1_std_units", "butterfly_violation_rate", "hedge20_std_ratio_wings",
            "hedge20_abs_mean_minus_oracle_bps"]
    print("| method | scenario | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for r in sorted(rows, key=lambda r: (r["scenario"], r.get("iv_rmse_heldout") or 1e9)):
        cells = [("" if r.get(c) is None else f"{r[c]:.3g}") for c in cols]
        print(f"| {r['method']} | {r['scenario']} | " + " | ".join(cells) + " |")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("score"); a.add_argument("--test", default="bench_data/D2/test.npz")
    a.add_argument("--pred", nargs="+", required=True); a.add_argument("--out", default="results/trackB")
    b = sub.add_parser("table"); b.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "score":
        for p in args.pred:
            score_and_save(args.test, p, args.out)
    else:
        table([p for g in args.paths for p in glob.glob(g)])


if __name__ == "__main__":
    main()
