"""Figures for the paper and presentation (static PNGs, matplotlib).

Each model keeps the same color in every figure (color follows the model, not its rank); truth / realized is
always near-black. Subcommands:

  paths       simulated price paths per model, starting at 100, like the team notebook's plot, plus 5-95% band,
              median, and the truth (synthetic) or what actually happened (real)
  terminal    terminal-return densities of all models vs truth on a log scale (where tail differences show)
  smile       synthetic: implied-vol smile per model vs truth for a calm and a stressed context
  leaderboard Track A: IV error by strike + grouped bars (ATM / wings / stressed wings)
  backtest    real data: CRPS, coverage, tail exceedances, PIT histograms, payoff-bias t-stats by strike
  market      model vs market implied-vol smiles per ticker (from bench.market compare results)
  trackB      Track B: smiles of one test surface across methods, densities, RMSE bars, Schroedinger-bridge errors

Examples:
  python -m bench.plots paths --data bench_data/A --samples runs/a6/samples.npy runs/a1/samples.npy --out figs
  python -m bench.plots leaderboard --results "results/trackA/*.json" --out figs
  python -m bench.plots backtest --results "results/backtest/*.json" --out figs
  python -m bench.plots trackB --test bench_data/D2/test.npz --preds results/trackB/preds --scenario sparse10 --out figs
"""
import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# fixed model -> slot assignment (color follows the model everywhere); unknown labels get the next free slot
ORDER = ["ar_diffusion", "ar_token", "a3_rnade", "a4_gan", "a5_sde_gan", "a6_pure_diffusion", "a7_path_flow"]
# classical baselines are neutral grays, told apart by line style / hatch (they are references, not contenders)
BASELINES = {"baseline_bsm": ("#8a8984", "--", "//"), "baseline_garch": ("#3d3c39", "-.", ".."),
             "baseline_bootstrap": ("#b5b4ae", ":", "xx")}
_assigned = {}


def color_for(label):
    for k, (c, _, _) in BASELINES.items():
        if label.startswith(k):
            return c
    for i, key in enumerate(ORDER):
        if label.startswith(key):
            return PALETTE[i]
    if label not in _assigned:
        _assigned[label] = PALETTE[(len(ORDER) + len(_assigned)) % len(PALETTE)]
    return _assigned[label]


def ls_for(label):
    for k, (_, ls, _) in BASELINES.items():
        if label.startswith(k):
            return ls
    return "-"


def hatch_for(label):
    for k, (_, _, h) in BASELINES.items():
        if label.startswith(k):
            return h
    return None


def style():
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 11, "axes.titlecolor": INK, "font.size": 9, "legend.frameon": False,
                         "lines.linewidth": 2.0, "savefig.dpi": 150, "savefig.bbox": "tight"})


def label_of(samples_path):
    mp = os.path.join(os.path.dirname(samples_path), "meta.json")
    if os.path.exists(mp):
        return json.load(open(mp)).get("label", os.path.basename(os.path.dirname(samples_path)))
    return os.path.basename(os.path.dirname(samples_path))


def pick_contexts(data, n=3):
    ctx = np.load(os.path.join(data, "eval_ctx.npy"))
    rv = np.sqrt((ctx.astype(np.float64) ** 2).mean(1))
    order = np.argsort(rv)
    idx = [order[0], order[len(order) // 2], order[-1]][:n]
    return idx, ["calmest context", "median context", "most stressed context"][:n]


def truth_paths(data):
    cfg = json.load(open(os.path.join(data, "config.json")))
    if cfg.get("kind", "synthetic") == "synthetic" and os.path.exists(os.path.join(data, "truth.npz")):
        t = np.load(os.path.join(data, "truth.npz"))
        if "truth_paths" in t.files:
            return "truth", t["truth_paths"]
    if os.path.exists(os.path.join(data, "eval_realized.npy")):
        return "realized", np.load(os.path.join(data, "eval_realized.npy"))[:, None, :]
    return None, None


# ---------------------------------------------------------------- paths (team-notebook style)
def plot_paths(a):
    style()
    idx, names = pick_contexts(a.data)
    ctx = np.load(os.path.join(a.data, "eval_ctx.npy")).astype(np.float64)
    tkind, tp = truth_paths(a.data)
    show_hist = 30
    for sp in a.samples:
        lab = label_of(sp)
        smp = np.load(sp).astype(np.float64)
        H = smp.shape[2]
        fig, axes = plt.subplots(1, len(idx), figsize=(4.2 * len(idx), 3.4), sharey=False)
        for ax, e, nm in zip(np.atleast_1d(axes), idx, names):
            hist = 100 * np.exp(np.cumsum(ctx[e, -show_hist:]) - ctx[e, -show_hist:].sum())  # ends at 100
            ax.plot(np.arange(-show_hist + 1, 1), hist, color=INK2, lw=1.4)
            P = 100 * np.exp(np.cumsum(smp[e], -1))
            P = np.concatenate([np.full((P.shape[0], 1), 100.0), P], 1)
            t = np.arange(0, H + 1)
            c = color_for(lab)
            for j in range(min(a.n_lines, P.shape[0])):
                ax.plot(t, P[j], color=c, lw=0.6, alpha=0.35)
            lo, med, hi = np.percentile(P, [5, 50, 95], axis=0)
            ax.fill_between(t, lo, hi, color=c, alpha=0.12, lw=0, label="model 5-95%")
            ax.plot(t, med, color=c, lw=2, label="model median")
            if tkind == "truth":
                Q = 100 * np.exp(np.cumsum(tp[e], -1)); Q = np.concatenate([np.full((Q.shape[0], 1), 100.0), Q], 1)
                tl, tm, th = np.percentile(Q, [5, 50, 95], axis=0)
                ax.plot(t, tl, color=INK, lw=1.2, ls="--", label="true 5-95%"); ax.plot(t, th, color=INK, lw=1.2, ls="--")
            elif tkind == "realized":
                R = 100 * np.exp(np.concatenate([[0], np.cumsum(tp[e, 0])]))
                ax.plot(t, R, color=INK, lw=2.2, label="what actually happened")
            ax.axvline(0, color=GRID, lw=1)
            b1, b99 = np.percentile(P, [5, 95], axis=0)
            ylo = min(b1.min(), hist.min()); yhi = max(b99.max(), hist.max())
            spread = yhi - ylo
            ylo, yhi = ylo - 0.25 * spread, yhi + 0.25 * spread
            if tkind == "realized":
                ylo, yhi = min(ylo, R.min()), max(yhi, R.max())
            pad = 0.05 * (yhi - ylo)
            ax.set_ylim(ylo - pad, yhi + pad)  # extreme samples (if any) are clipped from view, not from the data
            outside = np.mean((P.max(1) > yhi + pad) | (P.min(1) < ylo - pad))
            if outside > 0.005:
                ax.text(0.98, 0.02, f"{outside:.1%} of paths leave this view", transform=ax.transAxes, ha="right",
                        va="bottom", fontsize=8, color=INK2)
            ax.set_title(nm); ax.set_xlabel("trading days (0 = today)"); ax.set_ylabel("price")
        np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8)
        fig.suptitle(f"Simulated price paths: {lab}", color=INK, fontsize=12, y=1.04)
        os.makedirs(a.out, exist_ok=True)
        f = os.path.join(a.out, f"paths_{lab}.png"); fig.savefig(f); plt.close(fig); print("saved", f)


# ---------------------------------------------------------------- terminal densities
def plot_terminal(a):
    style()
    idx, names = pick_contexts(a.data)
    tkind, tp = truth_paths(a.data)
    S = {label_of(p): np.load(p).astype(np.float64) for p in a.samples}
    fig, axes = plt.subplots(1, len(idx), figsize=(4.4 * len(idx), 3.4))
    for ax, e, nm in zip(np.atleast_1d(axes), idx, names):
        allx = np.concatenate([s[e].sum(-1) for s in S.values()] + ([tp[e].sum(-1)] if tkind == "truth" else []))
        lo, hi = np.percentile(allx, [0.02, 99.98])
        grid = np.linspace(lo, hi, 400)
        if tkind == "truth":
            ax.plot(grid, np.maximum(gaussian_kde(tp[e].sum(-1))(grid), 1e-4), color=INK, lw=2.4, label="truth")
        for lab, s in S.items():
            x = s[e].sum(-1)
            x = x[np.isfinite(x)][:20000]
            ax.plot(grid, np.maximum(gaussian_kde(x)(grid), 1e-4), color=color_for(lab), ls=ls_for(lab), lw=1.5, label=lab)
        if tkind == "realized":
            ax.axvline(tp[e, 0].sum(), color=INK, lw=2, label="realized")
        ax.set_ylim(1e-3, None)
        ax.set_yscale("log"); ax.set_title(nm); ax.set_xlabel("20-day log return"); ax.set_ylabel("density (log scale)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 6), fontsize=8, bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("Terminal-return distributions, smoothed (tails are where option prices differ)", color=INK, fontsize=12, y=1.04)
    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, "terminal_distributions.png"); fig.savefig(f); plt.close(fig); print("saved", f)


# ---------------------------------------------------------------- smiles (synthetic)
def plot_smile(a):
    from ardiff.evaluate import implied_vol, otm_prices
    style()
    cfg = json.load(open(os.path.join(a.data, "config.json")))
    tr = np.load(os.path.join(a.data, "truth.npz"))
    ks = np.array(cfg["ks"]); H = cfg["horizon"]
    sd_H = cfg["garch"]["annual_vol"] * math.sqrt(H / 252)
    sv = tr["state_vol"]
    idx = [int(np.argmin(sv)), int(np.argmax(sv))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, e, nm in zip(axes, idx, ["calmest context", "most stressed context"]):
        ax.plot(ks, tr["iv"][e] * 100, color=INK, lw=2.4, label="truth")
        for p in a.samples:
            lab = label_of(p); s = np.load(p).astype(np.float64)
            P, _ = otm_prices(s[e:e + 1].sum(-1), ks, sd_H)
            ax.plot(ks, implied_vol(P, ks, sd_H, H / 252)[0] * 100, color=color_for(lab), ls=ls_for(lab), lw=1.6, marker="o", ms=4, label=lab)
        ax.set_title(f"{nm} (state vol {sv[e]:.0%})"); ax.set_xlabel("strike, std units of 20-day vol (puts < 0 < calls)")
        ax.set_ylabel("implied vol (%)")
    axes[1].legend(fontsize=7)
    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, "smiles.png"); fig.savefig(f); plt.close(fig); print("saved", f)


# ---------------------------------------------------------------- Track A leaderboard
def plot_leaderboard(a):
    style()
    rows = [json.load(open(p)) for g in a.results for p in glob.glob(g)]
    rows = sorted(rows, key=lambda r: r["iv_err_wings"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1.2, 1]))
    for r in rows:
        axes[0].plot(r["ks"], r["mean_abs_iv_err_by_k"], color=color_for(r["label"]), ls=ls_for(r["label"]), marker="o", ms=4, lw=1.6, label=r["label"])
    axes[0].set_xlabel("strike, std units (puts < 0 < calls)"); axes[0].set_ylabel("mean |IV error| (vol pts)")
    axes[0].set_title("Pricing error by strike"); axes[0].legend(fontsize=7)
    cols = [("iv_err_atm", "ATM"), ("iv_err_wings", "wings"), ("iv_err_wings_stressed", "wings, stressed")]
    x = np.arange(len(cols)); w = 0.8 / max(len(rows), 1)
    for i, r in enumerate(rows):
        axes[1].bar(x + i * w - 0.4 + w / 2, [r[c] for c, _ in cols], w * 0.9, color=color_for(r["label"]),
                    hatch=hatch_for(r["label"]), edgecolor="white", label=r["label"])
    axes[1].set_xticks(x, [n for _, n in cols]); axes[1].set_ylabel("mean |IV error| (vol pts)"); axes[1].set_title("Summary")
    axes[1].grid(axis="x", visible=False)
    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, "trackA_leaderboard.png"); fig.savefig(f); plt.close(fig); print("saved", f)


# ---------------------------------------------------------------- real-data backtest
def plot_backtest(a):
    style()
    rows = sorted([json.load(open(p)) for g in a.results for p in glob.glob(g)], key=lambda r: r["crps_mean"])
    n = len(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    names = [r["label"] for r in rows]; cols = [color_for(l) for l in names]
    axes[0].barh(names, [r["crps_mean"] for r in rows], color=cols, hatch=[hatch_for(l) for l in names], edgecolor="white")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("CRPS of 20-day return (lower is better)"); axes[0].set_title("Forecast accuracy"); axes[0].grid(axis="y", visible=False)
    y = np.arange(n)
    axes[1].scatter([r["coverage90"] for r in rows], y, color=cols, s=60, zorder=3, label="90% interval coverage")
    axes[1].scatter([r["exceed_5pct"] * 10 for r in rows], y, color=cols, s=60, marker="s", zorder=3, label="5% tail exceedance x10")
    axes[1].axvline(0.9, color=INK, lw=1, ls="--"); axes[1].axvline(0.5, color=INK2, lw=1, ls=":")
    axes[1].set_yticks(y, names); axes[1].invert_yaxis(); axes[1].set_xlim(0, 1.05)
    axes[1].set_title("Calibration (targets: 0.90 and 0.5)"); axes[1].legend(fontsize=7, loc="lower left")
    for r in rows:
        axes[2].plot(r["ks"], r["payoff_bias_t"], color=color_for(r["label"]), ls=ls_for(r["label"]), marker="o", ms=4, lw=1.5, label=r["label"])
    axes[2].axhspan(-2, 2, color=GRID, alpha=0.6, lw=0)
    axes[2].set_xlabel("strike, std units of context vol"); axes[2].set_ylabel("t-stat of (model - realized) payoff")
    axes[2].set_title("Payoff bias vs what happened (|t| < 2 is fine)"); axes[2].legend(fontsize=7)
    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, "backtest_summary.png"); fig.savefig(f); plt.close(fig); print("saved", f)
    cols_n = min(4, n); rws = math.ceil(n / cols_n)
    fig, axes = plt.subplots(rws, cols_n, figsize=(3.2 * cols_n, 2.6 * rws), squeeze=False)
    for ax, r in zip(axes.ravel(), rows):
        ax.hist(r["pit"], bins=10, range=(0, 1), color=color_for(r["label"]), edgecolor="white", linewidth=2)
        ax.axhline(len(r["pit"]) / 10, color=INK, ls="--", lw=1)
        ax.set_title(r["label"], fontsize=9); ax.set_xlabel("PIT (flat = calibrated)")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    f = os.path.join(a.out, "backtest_pit.png"); fig.savefig(f); plt.close(fig); print("saved", f)


# ---------------------------------------------------------------- market smiles
def plot_market(a):
    style()
    rows = [json.load(open(p)) for g in a.results for p in glob.glob(g)]
    tick = sorted(set(t for r in rows for t in r["tickers"]))
    if not tick:
        print("no market results"); return
    cols_n = min(3, len(tick)); rws = math.ceil(len(tick) / cols_n)
    fig, axes = plt.subplots(rws, cols_n, figsize=(4.4 * cols_n, 3.3 * rws), squeeze=False)
    for ax, t in zip(axes.ravel(), tick):
        first = True
        for r in rows:
            if t not in r["tickers"]:
                continue
            d = r["tickers"][t]
            x = np.array(d["log_moneyness"]); o = np.argsort(x)
            if first:
                ax.scatter(x[o], np.array(d["iv_market"])[o] * 100, color=INK, s=14, zorder=3, label="market mid")
                ax.axhline(d["ctx_vol"] * 100, color=INK2, ls=":", lw=1, label="realized vol (BSM)")
                first = False
            ax.plot(x[o], np.array(d["iv_model_Q"])[o] * 100, color=color_for(r["label"]), ls=ls_for(r["label"]), lw=1.6, label=r["label"])
        ax.set_title(f"{t}  (expiry {d['expiry']})"); ax.set_xlabel("log(K / forward)"); ax.set_ylabel("implied vol (%)")
    axes.ravel()[0].legend(fontsize=7)
    for ax in axes.ravel()[len(tick):]:
        ax.axis("off")
    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, "market_smiles.png"); fig.savefig(f); plt.close(fig); print("saved", f)


# ---------------------------------------------------------------- Track B
def plot_trackB(a):
    style()
    te = dict(np.load(a.test))
    sc = a.scenario
    preds = {}
    for p in sorted(glob.glob(os.path.join(a.preds, f"*_{sc}.npz"))):
        d = np.load(p)
        if "surfaces" in d.files:
            continue
        preds[str(d["method"])] = {k: d[k] for k in d.files}
    s = a.surface
    ks, taus_d = te["ks"], te["taus_days"]
    mask, quote = te[f"mask_{sc}"][s], te[f"quote_{sc}"][s]
    classical = [m for m in preds if m in ("linear2d", "ssvi", "gp", "heston", "maxent")]
    neural = [m for m in preds if m not in classical]
    groups = [(n, g) for n, g in (("classical", classical), ("neural", neural)) if g]
    fig, axes = plt.subplots(len(groups), len(taus_d), figsize=(3.2 * len(taus_d), 3.0 * len(groups)), sharey=True, squeeze=False)
    for row, (gname, meths) in enumerate(groups):
        for j, ax in enumerate(axes[row]):
            ax.plot(ks, te["iv"][s, j] * 100, color=INK, lw=2.4, label="truth")
            for i, m in enumerate(meths):
                d = preds[m]
                ax.plot(ks, d["iv"][s, j] * 100, color=PALETTE[i % 8], lw=1.5, label=m)
                if "iv_lo" in d:
                    ax.fill_between(ks, d["iv_lo"][s, j] * 100, d["iv_hi"][s, j] * 100, color=PALETTE[i % 8], alpha=0.12, lw=0)
            ax.scatter(ks[mask[j]], quote[j][mask[j]] * 100, color=INK, marker="x", s=30, zorder=4, label="quotes")
            ax.set_title(f"{taus_d[j]} days" if row == 0 else ""); ax.set_xlabel("strike (std units)" if row == len(groups) - 1 else "")
        axes[row, 0].set_ylabel(f"{gname}: implied vol (%)")
        axes[row, -1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"Surface completion, test surface {s} ({te['family'][s]}), scenario {sc}", color=INK, fontsize=12, y=1.02)
    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, f"trackB_smiles_{sc}_s{s}.png"); fig.savefig(f); plt.close(fig); print("saved", f)

    g = [i for i in range(len(te["family"])) if te["family"][i] == "garch"]
    dens = {m: d["density"] for m, d in preds.items() if "density" in d}
    if dens and g:
        sg = g[0]; j = list(taus_d).index(20)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        z = te["zgrid"]
        ax.plot(z, np.maximum(te["density"][sg, j], 1e-5), color=INK, lw=2.4, label="truth")
        for i, (m, d) in enumerate(dens.items()):
            ax.plot(z, np.maximum(d[sg, j], 1e-5), color=PALETTE[list(preds).index(m) % 8], lw=1.4, label=m)
        ax.set_yscale("log"); ax.set_xlim(-8, 8); ax.set_ylim(1e-5, None)
        ax.set_xlabel("20-day log return (std units)"); ax.set_ylabel("density (log scale)")
        ax.set_title(f"Risk-neutral density, surface {sg}, scenario {sc}"); ax.legend(fontsize=7)
        f = os.path.join(a.out, f"trackB_density_{sc}.png"); fig.savefig(f); plt.close(fig); print("saved", f)

    res = [json.load(open(p)) for p in glob.glob(os.path.join(a.results, "*.json"))]
    res = [r for r in res if "iv_rmse_heldout" in r and r.get("scenario") in ("per_mat5", "sparse10", "nowings")]
    if res:
        meths = sorted(set(r["method"] for r in res), key=lambda m: np.mean([r["iv_rmse_heldout"] for r in res if r["method"] == m]))
        scs = ["per_mat5", "sparse10", "nowings"]
        fig, ax = plt.subplots(figsize=(9, 3.6))
        w = 0.8 / len(meths); x = np.arange(len(scs))
        for i, m in enumerate(meths):
            v = [next((r["iv_rmse_heldout"] for r in res if r["method"] == m and r["scenario"] == c), np.nan) for c in scs]
            ax.bar(x + i * w - 0.4 + w / 2, v, w * 0.9, color=PALETTE[i % 8], label=m)
        ax.set_xticks(x, ["5 quotes per maturity", "10 quotes total", "no wings"]); ax.set_ylabel("held-out IV RMSE (vol pts)")
        ax.set_title("Track B: surface completion error"); ax.legend(fontsize=7, ncol=2); ax.grid(axis="x", visible=False)
        f = os.path.join(a.out, "trackB_rmse.png"); fig.savefig(f); plt.close(fig); print("saved", f)

    sb = glob.glob(os.path.join(a.results, "b4_sb_*.json"))
    if sb:
        d = json.load(open(sb[0]))["summary"]
        pay = list(next(iter(d.values())).keys())
        fig, ax = plt.subplots(figsize=(8, 3.4))
        meths = list(d); w = 0.8 / len(meths); x = np.arange(len(pay))
        for i, m in enumerate(meths):
            ax.bar(x + i * w - 0.4 + w / 2, [d[m][p] * 100 for p in pay], w * 0.9, color=PALETTE[i % 8], label=m)
        ax.set_xticks(x, ["fwd-start straddle", "fwd-start 90% put", "fwd-start 110% call"][:len(pay)])
        ax.set_ylabel("mean |relative error| (%)"); ax.set_title("B4: joining maturities (Schroedinger bridge vs alternatives)")
        ax.legend(fontsize=7); ax.grid(axis="x", visible=False)
        f = os.path.join(a.out, "trackB_bridge.png"); fig.savefig(f); plt.close(fig); print("saved", f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("paths", "terminal", "smile"):
        p = sub.add_parser(name); p.add_argument("--data", required=True); p.add_argument("--samples", nargs="+", required=True)
        p.add_argument("--out", default="figs"); p.add_argument("--n_lines", type=int, default=60)
    for name in ("leaderboard", "backtest", "market"):
        p = sub.add_parser(name); p.add_argument("--results", nargs="+", required=True); p.add_argument("--out", default="figs")
    p = sub.add_parser("trackB")
    p.add_argument("--test", required=True); p.add_argument("--preds", default="results/trackB/preds")
    p.add_argument("--results", default="results/trackB"); p.add_argument("--scenario", default="sparse10")
    p.add_argument("--surface", type=int, default=0); p.add_argument("--out", default="figs")
    a = ap.parse_args(argv)
    dict(paths=plot_paths, terminal=plot_terminal, smile=plot_smile, leaderboard=plot_leaderboard,
         backtest=plot_backtest, market=plot_market, trackB=plot_trackB)[a.cmd](a)


if __name__ == "__main__":
    main()
