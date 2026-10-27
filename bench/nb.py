"""Helpers for notebooks/run_all.ipynb: run models from a notebook with live logs, live loss curves and a
leaderboard + comparison figures that refresh as each model finishes.

Everything here calls the same command-line entry points as the shell scripts (scripts/run_model.sh, the Track B
starters, bench.plots), so a notebook run and a PACE batch run produce identical files:
  logs/notebook/<data name>/<model>.log    full stdout/stderr of each run
  runs/, results/, figs/                   exactly as scripts/run_model.sh and scripts/report.sh write them

Two modes:
  local  run_trackA / run_trackB execute inside the notebook's own session (an OnDemand Jupyter on a GPU node).
  slurm  submit_trackA sends one sbatch job per model (scripts/submit_all.sh); watch() polls squeue and redraws the
         loss curves of every job from its log file plus the leaderboard, until all jobs finish.
"""
import glob
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bench.plots import color_for, ls_for, style  # noqa: E402

try:
    from IPython.display import HTML, Image, Markdown, display
except ImportError:  # plain python: print instead
    display = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRACK_A = ["bsm", "bootstrap", "garch", "ar_diffusion", "ar_token", "rnade", "gan", "sde_gan",
           "pure_diffusion", "path_flow", "path_flow_t"]
SETS = {
    "baselines": ["bsm", "bootstrap", "garch"],
    "thesis": ["ar_diffusion", "ar_token"],                       # A1 vs A2: the head-to-head the paper is about
    "generative": ["rnade", "gan", "sde_gan", "pure_diffusion", "path_flow", "path_flow_t"],
    "all": TRACK_A,
}
# Track B: name -> (module, needs --train, quick-mode extra args)
TRACK_B = {
    "b_baselines": ("starters.b_baselines", False, []),
    "b1_ebm": ("starters.b1_ebm", True, ["--width", "64", "--steps", "300", "--batch", "32", "--log_every", "150"]),
    "b1_flow": ("starters.b1_flow", False, ["--steps", "300"]),
    "b3_neural_process": ("starters.b3_neural_process", True, ["--width", "64", "--steps", "500", "--batch", "32", "--log_every", "250"]),
    "b3_vae": ("starters.b3_vae", True, ["--steps", "1000", "--log_every", "500", "--fit_steps", "100"]),
    "b2_neural_sde": ("starters.b2_neural_sde", False, ["--n_surfaces", "2", "--ensemble", "2", "--width", "32", "--paths", "512",
                                                        "--price_paths", "5000", "--steps", "50"]),
    "b4_schrodinger_bridge": ("starters.b4_schrodinger_bridge", False, ["--n_surfaces", "4"]),
}

_STEP = re.compile(r"^\s*step\s+(\d+)\s+(.*)$")
_PAIR = re.compile(r"([A-Za-z][A-Za-z0-9/ \-]*?)\s+(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)(?=\s|$)")


# ------------------------------------------------------------------------------------------------ parsing
def parse_step(line):
    """'step 500  train 0.41  val 0.43  12s' -> (500, {'train': 0.41, 'val': 0.43}). Works for every starter's log."""
    m = _STEP.match(line)
    if not m:
        return None
    rest = re.sub(r"\(.*?\)", " ", m.group(2))
    rest = re.sub(r"\s\d+(?:\.\d+)?s\s*$", "", " " + rest)
    vals = {k.strip(): float(v) for k, v in _PAIR.findall(rest) if k.strip()}
    return (int(m.group(1)), vals) if vals else None


def parse_log(path):
    hist = {}
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            p = parse_step(line)
            if p:
                for k, v in p[1].items():
                    hist.setdefault(k, []).append((p[0], v))
    return hist


def _panels(hist):
    """Group 'train X' and 'val X' series on one panel; everything else gets its own panel (max 4)."""
    panels = {}
    for k in hist:
        base = re.sub(r"^(train|val)\s*", "", k) or "loss"
        panels.setdefault(base, []).append(k)
    return dict(list(panels.items())[:4])


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def loss_figure(hists, title=""):
    """hists: {model: {series: [(step, value)]}} -> PNG bytes, one row of panels per model."""
    style()
    hists = {m: h for m, h in hists.items() if h}
    if not hists:
        return None
    n = len(hists)
    ncol = max(len(_panels(h)) for h in hists.values())
    fig, axes = plt.subplots(n, ncol, figsize=(3.6 * ncol, 2.4 * n), squeeze=False)
    for i, (m, h) in enumerate(hists.items()):
        for j in range(ncol):
            axes[i, j].set_visible(False)
        for j, (base, keys) in enumerate(_panels(h).items()):
            ax = axes[i, j]; ax.set_visible(True)
            for k in keys:
                s, v = np.array(h[k]).T
                ax.plot(s, v, "-" if not k.startswith("val") else "--", lw=1.6, color=color_for(m) if not k.startswith("val") else "#52514e",
                        label=k.split()[0] if k.split()[0] in ("train", "val") else None)
            vv = np.concatenate([np.array(h[k])[:, 1] for k in keys])
            if np.all(vv > 0) and vv.max() / max(vv.min(), 1e-12) > 30:
                ax.set_yscale("log")
            ax.set_title(f"{m}: {base}", fontsize=9, loc="left")
            ax.tick_params(labelsize=7)
            if any(k.split()[0] in ("train", "val") for k in keys):
                ax.legend(fontsize=7, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return _png(fig)


# ------------------------------------------------------------------------------------------------ display
class _Live:
    """A status line, a log tail and a figure that update in place in the notebook output."""

    def __init__(self, title):
        self.title = title
        if display:
            self.h_status = display(Markdown(f"**{title}**: starting"), display_id=True)
            self.h_fig = display(HTML(""), display_id=True)
            self.h_log = display(HTML(""), display_id=True)

    def status(self, text):
        if display:
            self.h_status.update(Markdown(f"**{self.title}**: {text}"))
        else:
            print(f"[{self.title}] {text}", flush=True)

    def log(self, lines):
        if display:
            txt = "\n".join(lines).replace("&", "&amp;").replace("<", "&lt;")
            self.h_log.update(HTML(f"<pre style='font-size:11px;line-height:1.25;max-height:260px;overflow:auto;"
                                   f"background:#f6f6f4;padding:6px'>{txt}</pre>"))

    def fig(self, png):
        if display and png:
            self.h_fig.update(Image(data=png))


def _stream(cmd, log_path, title, env=None, refresh=2.0, tail=18):
    """Run cmd, tee to log_path, show the last lines + live loss curves. Returns (returncode, seconds)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    live = _Live(title)
    live.status(f"running  `{' '.join(cmd) if isinstance(cmd, list) else cmd}`")
    e = dict(os.environ, PYTHONUNBUFFERED="1", **(env or {}))
    t0 = time.time(); last = 0.0
    lines, hist = [], {}
    with open(log_path, "w") as f:
        f.write(f"# {time.strftime('%F %T')}  {cmd}\n")
        p = subprocess.Popen(cmd, cwd=ROOT, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             shell=isinstance(cmd, str))
        for line in p.stdout:
            f.write(line); f.flush()
            line = line.rstrip("\n")
            lines.append(line); lines = lines[-tail:]
            if not display:
                print(line, flush=True)
            st = parse_step(line)
            if st:
                for k, v in st[1].items():
                    hist.setdefault(k, []).append((st[0], v))
            if time.time() - last > refresh:
                live.log(lines); last = time.time()
                if st:
                    live.fig(loss_figure({title: hist}))
                live.status(f"running, {time.time() - t0:.0f}s" + (f", step {st[0]}" if st else ""))
        rc = p.wait()
    live.log(lines)
    live.fig(loss_figure({title: hist}))
    dt = time.time() - t0
    live.status(("done" if rc == 0 else f"**FAILED (exit {rc})**, see `{os.path.relpath(log_path, ROOT)}`") + f" in {dt:.0f}s")
    return rc, dt


def sh(cmd, title=None, log=None):
    """Run any shell command with a live log (e.g. data export)."""
    log = log or os.path.join(ROOT, "logs", "notebook", "misc", re.sub(r"\W+", "_", title or cmd)[:60] + ".log")
    return _stream(cmd, log, title or cmd)[0]


# ------------------------------------------------------------------------------------------------ data
def ensure_data(synthetic="bench_data/A", d2="bench_data/D2", n_hedge_paths=None, quick=False):
    """Create the synthetic Track A export and the Track B surfaces if they are missing (skips existing ones)."""
    if synthetic and not os.path.exists(os.path.join(ROOT, synthetic, "config.json")):
        extra = " --n_eval 8 --n_mc 20000" if quick else ""
        if n_hedge_paths:
            extra += f" --n_hedge_paths {n_hedge_paths}"
        sh(f"{sys.executable} -m bench.trackA export --out {synthetic}{extra}", "export synthetic Track A data")
    elif synthetic:
        print(f"ok: {synthetic}")
    if d2 and not os.path.exists(os.path.join(ROOT, d2, "test.npz")):
        extra = " --n_train 64 --n_test 12 --paths 5000 --test_paths 30000" if quick else ""
        sh(f"{sys.executable} -m bench.surfaces --out {d2}{extra}", "build Track B surfaces (D2)")
    elif d2:
        print(f"ok: {d2}")


def data_info(data):
    cfg = json.load(open(os.path.join(ROOT, data, "config.json")))
    tr = np.load(os.path.join(ROOT, data, "train.npy"), mmap_mode="r")
    ev = np.load(os.path.join(ROOT, data, "eval_ctx.npy"), mmap_mode="r")
    return cfg, tr, ev


def show_data(data, n=40):
    """Quick look at a Track A data folder: sample training windows as price paths + return histogram."""
    cfg, tr, ev = data_info(data)
    style()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(tr), size=min(n, len(tr)), replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    paths = 100 * np.exp(np.cumsum(np.asarray(tr[idx]), axis=1))
    L = cfg["ctx_len"]
    ax[0].plot(paths.T, lw=0.7, alpha=0.6, color="#2a78d6")
    ax[0].axvline(L - 0.5, color="#0b0b0b", lw=1, ls=":")
    ax[0].set_title(f"{n} training windows ({L} context days | {cfg['horizon']} future days)", fontsize=10, loc="left")
    r = np.asarray(tr[:5000]).ravel()
    ax[1].hist(r / r.std(), bins=200, color="#2a78d6", alpha=0.8, log=True)
    ax[1].set_title(f"daily returns / std (kurtosis {((r - r.mean())**4).mean() / r.var()**2:.1f})", fontsize=10, loc="left")
    fig.tight_layout()
    if display:
        display(Markdown(f"**{data}**: kind `{cfg.get('kind', 'synthetic')}`, train {tr.shape}, eval contexts {ev.shape}"
                         + (f", tickers {cfg.get('tickers')}" if cfg.get("tickers") else "")))
        display(Image(data=_png(fig)))
    else:
        plt.close(fig)


# ------------------------------------------------------------------------------------------------ results
def _name(data):
    return os.path.basename(os.path.normpath(data))


def _kind(data):
    return json.load(open(os.path.join(ROOT, data, "config.json"))).get("kind", "synthetic")


def results_table(data):
    """pandas DataFrame of every scored model on this data folder (Track A or backtest), plus market columns."""
    import pandas as pd
    name = _name(data)
    sub = "trackA" if _kind(data) == "synthetic" else "backtest"
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", name, sub, "*.json"))):
        r = json.load(open(p))
        rows.append({k: v for k, v in r.items() if isinstance(v, (int, float, str)) or v is None})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.set_index("label")
    for p in glob.glob(os.path.join(ROOT, "results", name, "market", "*.json")):
        r = json.load(open(p))
        for k in ("mean_abs_iv_err_Q", "mean_abs_iv_err_bsm_ctxvol"):
            if r.get("label") in df.index:
                df.loc[r["label"], "market_" + k] = r.get(k)
    key = "iv_err_wings" if sub == "trackA" else "crps_mean"
    first = [c for c in (["iv_err_atm", "iv_err_wings", "iv_err_wings_stressed", "asian_rel_err", "upout_rel_err",
                          "explosive_rate", "acf_abs_lag1_model", "acf_abs_lag1_true"] if sub == "trackA"
                         else ["crps_mean", "coverage90", "exceed_1pct", "exceed_5pct", "qlike"]) if c in df]
    rest = [c for c in df.columns if c not in first and c != "n_paths"]
    return df[first + rest].sort_values(key) if key in df else df


def comparison_figure(data):
    """Track A: IV error by strike (all models so far) + ATM / wings bars. Real data: CRPS + coverage bars."""
    style()
    name = _name(data)
    if _kind(data) == "synthetic":
        rs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(ROOT, "results", name, "trackA", "*.json")))]
        if not rs:
            return None
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
        for r in rs:
            ax[0].plot(r["ks"], r["mean_abs_iv_err_by_k"], ls_for(r["label"]), color=color_for(r["label"]), lw=1.8, label=r["label"])
        ax[0].set_xlabel("log-moneyness (std units)"); ax[0].set_ylabel("mean |IV error| (vol pts)")
        ax[0].set_title("IV error by strike: the wings are where the thesis lives", fontsize=10, loc="left")
        ax[0].legend(fontsize=7, frameon=False, ncol=2)
        rs = sorted(rs, key=lambda r: r["iv_err_wings"])
        x = np.arange(len(rs)); w = 0.4
        for j, (k, a) in enumerate((("iv_err_atm", 0.45), ("iv_err_wings", 1.0))):
            ax[1].bar(x + (j - 0.5) * w, [r[k] for r in rs], w, color=[color_for(r["label"]) for r in rs], alpha=a,
                      label=k.replace("iv_err_", ""), edgecolor="white")
        ax[1].set_xticks(x); ax[1].set_xticklabels([r["label"] for r in rs], rotation=45, ha="right", fontsize=7)
        ax[1].set_title("ATM (light) vs wings (solid), lower is better", fontsize=10, loc="left")
        # one diverged model must not flatten everyone else: cap at 3x the median model, list who is off scale
        peak = {r["label"]: max(np.nanmax(r["mean_abs_iv_err_by_k"]), r["iv_err_atm"], r["iv_err_wings"]) for r in rs}
        cap = 3 * float(np.median(list(peak.values())))
        off = [k for k, v in peak.items() if v > cap]
        if off and cap > 0:
            for a in ax:
                a.set_ylim(0, cap)
            ax[0].set_xlabel("log-moneyness (std units)\noff scale (peak vol pts): " + ", ".join(f"{k} ({peak[k]:.0f})" for k in off),
                             color="#e34948")
    else:
        rs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(ROOT, "results", name, "backtest", "*.json")))]
        if not rs:
            return None
        rs = sorted(rs, key=lambda r: r["crps_mean"])
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
        x = np.arange(len(rs)); c = [color_for(r["label"]) for r in rs]
        for a, k, ttl, ref in ((ax[0], "crps_mean", "CRPS (lower better)", None), (ax[1], "coverage90", "90% interval coverage", 0.9),
                               (ax[2], "exceed_1pct", "1% tail exceedance rate", 0.01)):
            a.bar(x, [r[k] for r in rs], color=c)
            if ref is not None:
                a.axhline(ref, color="#0b0b0b", lw=1, ls=":")
            a.set_xticks(x); a.set_xticklabels([r["label"] for r in rs], rotation=45, ha="right", fontsize=7)
            a.set_title(ttl, fontsize=10, loc="left")
    fig.tight_layout()
    return _png(fig)


class Dashboard:
    """Leaderboard + comparison figure for one data folder; call .refresh() after each model."""

    def __init__(self, data):
        self.data = data
        if display:
            self.h_md = display(Markdown(f"### Leaderboard: `{data}`"), display_id=True)
            self.h_tab = display(HTML("<i>no results yet</i>"), display_id=True)
            self.h_fig = display(HTML(""), display_id=True)

    def refresh(self, note=""):
        df = results_table(self.data)
        if not display:
            print(df.to_string() if len(df) else "no results yet")
            return df
        self.h_md.update(Markdown(f"### Leaderboard: `{self.data}`  ({len(df)} models{', ' + note if note else ''})"))
        if len(df):
            num = df.select_dtypes("number").columns
            self.h_tab.update(HTML(df.style.format("{:.4g}", subset=num, na_rep="").to_html()))
        png = comparison_figure(self.data)
        if png:
            self.h_fig.update(Image(data=png))
        return df


# ------------------------------------------------------------------------------------------------ local runs
def _expand(models):
    out = []
    for m in ([models] if isinstance(models, str) else models):
        out += SETS.get(m, [m])
    return out


def run_trackA(models, data="bench_data/A", market=None, quick=False, n_paths=None, extra=None, dashboard=None,
               stop_on_error=False):
    """Train + sample + score + plot each model in turn (same as bash scripts/run_model.sh <model> <data>).

    models: a name ('pure_diffusion'), a set name ('baselines', 'thesis', 'generative', 'all') or a list mixing both.
    extra:  {model: '--steps 20000 --lr 3e-4'} extra args per model.
    """
    models = _expand(models)
    dash = dashboard or Dashboard(data)
    env = {}
    if market:
        env["MARKET"] = market
    if quick:
        env["QUICK"] = "1"
    if n_paths:
        env["N_PATHS"] = str(n_paths)
    summary = []
    for i, m in enumerate(models):
        if m not in TRACK_A:
            print(f"skip unknown model {m}; choose from {TRACK_A}")
            continue
        cmd = ["bash", "scripts/run_model.sh", m, data] + shlex.split((extra or {}).get(m, ""))
        log = os.path.join(ROOT, "logs", "notebook", _name(data), f"{m}.log")
        rc, dt = _stream(cmd, log, f"[{i + 1}/{len(models)}] {m} on {_name(data)}", env=env)
        summary.append((m, rc, dt))
        dash.refresh(f"last: {m}")
        fig = os.path.join(ROOT, "figs", _name(data), f"paths_{_label_of(data, m)}.png")
        if rc == 0 and display and os.path.exists(fig):
            display(Image(filename=fig, width=900))
        if rc != 0 and stop_on_error:
            break
    _print_summary(summary)
    return dash


def _label_of(data, model):
    mp = os.path.join(ROOT, "runs", _name(data), model, "meta.json")
    try:
        return json.load(open(mp))["label"]
    except Exception:
        return model


def _print_summary(summary):
    if display and summary:
        display(Markdown("| model | status | time |\n|---|---|---|\n" + "\n".join(
            f"| {m} | {'ok' if rc == 0 else 'FAILED (' + str(rc) + ')'} | {dt / 60:.1f} min |" for m, rc, dt in summary)))


def run_trackB(methods="all", d2="bench_data/D2", quick=False, extra=None):
    """Track B starters on the D2 option surfaces, then the Track B table and figures."""
    methods = list(TRACK_B) if methods == "all" else ([methods] if isinstance(methods, str) else methods)
    test, train = f"{d2}/test.npz", f"{d2}/train.npz"
    summary = []
    for i, m in enumerate(methods):
        mod, needs_train, q = TRACK_B[m]
        cmd = [sys.executable, "-m", mod, "--test", test] + (["--train", train] if needs_train else [])
        cmd += (q if quick else []) + shlex.split((extra or {}).get(m, ""))
        rc, dt = _stream(cmd, os.path.join(ROOT, "logs", "notebook", "trackB", f"{m}.log"), f"[{i + 1}/{len(methods)}] {m}")
        summary.append((m, rc, dt))
    _print_summary(summary)
    show_trackB(d2)


def trackB_table():
    import pandas as pd
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(ROOT, "results", "trackB", "*.json")))]
    rows = [{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in rows if "scenario" in r and "method" in r]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.pivot_table(index="method", columns="scenario", values="iv_rmse_heldout").round(3)


def b4_table():
    import pandas as pd
    p = os.path.join(ROOT, "results", "trackB", "b4_sb_true_marginals.json")
    if not os.path.exists(p):
        return pd.DataFrame()
    s = json.load(open(p))["summary"]
    return (pd.DataFrame(s).T * 100).round(1).rename(columns=lambda c: c.replace("fwd_", "").replace("_20_60", "") + " %err")


def show_trackB(d2="bench_data/D2", scenario="sparse10"):
    sh(f"{sys.executable} -m bench.plots trackB --test {d2}/test.npz --scenario {scenario} --out figs/trackB", "Track B figures")
    if display:
        display(Markdown("**Track B: held-out IV RMSE (vol pts) by method and scenario**"))
        display(trackB_table())
        b4 = b4_table()
        if len(b4):
            display(Markdown("**B4 Schroedinger bridge: mean |relative error| of forward-start prices**"))
            display(b4)
        for f in sorted(glob.glob(os.path.join(ROOT, "figs", "trackB", "*.png"))):
            display(Markdown(f"`{os.path.relpath(f, ROOT)}`")); display(Image(filename=f, width=900))


def report(data):
    """scripts/report.sh (tables + all comparison figures), then show every figure for this data folder."""
    sh(f"bash scripts/report.sh {data}", f"report {data}")
    show_figs(data)


def show_figs(data, pattern="*.png"):
    for f in sorted(glob.glob(os.path.join(ROOT, "figs", _name(data), pattern))):
        if display:
            display(Markdown(f"`{os.path.relpath(f, ROOT)}`")); display(Image(filename=f, width=900))


# ------------------------------------------------------------------------------------------------ slurm
def submit_trackA(models, data="bench_data/A", market=None):
    """One sbatch job per model + a report job (scripts/submit_all.sh). Returns {model: job id}."""
    env = dict(os.environ, MODELS=" ".join(_expand(models)))
    if market:
        env["MARKET"] = market
    out = subprocess.run(["bash", "scripts/submit_all.sh", data], cwd=ROOT, env=env, capture_output=True, text=True)
    print(out.stdout, out.stderr)
    jobs = dict(re.findall(r"^(\S+) -> job (\d+)", out.stdout, re.M))
    rep = re.search(r"report job (\d+)", out.stdout)
    if rep:
        jobs["report"] = rep.group(1)
    return jobs


def _job_log(name, data, jid):
    hits = glob.glob(os.path.join(ROOT, "logs", f"{name}_{_name(data)}-{jid}.out")) or glob.glob(os.path.join(ROOT, "logs", f"*-{jid}.out"))
    return hits[0] if hits else None


def watch(jobs, data="bench_data/A", every=30, max_hours=12):
    """Poll squeue; redraw every job's loss curves (from its log) and the leaderboard until all jobs leave the queue."""
    dash = Dashboard(data)
    live = _Live(f"PACE jobs for {_name(data)}")
    t0 = time.time()
    while True:
        q = subprocess.run(["squeue", "-h", "-j", ",".join(jobs.values()), "-o", "%i %T %M"], capture_output=True, text=True).stdout
        state = {l.split()[0]: (l.split()[1], l.split()[2]) for l in q.strip().splitlines() if l.strip()}
        rows, hists, tails = [], {}, []
        for m, jid in jobs.items():
            st, el = state.get(jid, ("finished", ""))
            log = _job_log(m, data, jid)
            rows.append(f"| {m} | {jid} | {st} | {el} | {os.path.relpath(log, ROOT) if log else ''} |")
            if log and m != "report":
                hists[m] = parse_log(log)
                if st == "RUNNING":
                    tails += [f"--- {m} ---"] + open(log, errors="replace").read().splitlines()[-4:]
        live.status(f"{len(state)} of {len(jobs)} jobs still queued/running, checked {time.strftime('%T')}\n\n"
                    "| model | job | state | elapsed | log |\n|---|---|---|---|---|\n" + "\n".join(rows))
        live.fig(loss_figure(hists, "training curves of every job (from logs/)"))
        live.log(tails[-40:])
        dash.refresh(f"updated {time.strftime('%T')}")
        if not state or time.time() - t0 > max_hours * 3600:
            break
        time.sleep(every)
    live.status("all jobs finished" if not state else "stopped watching (jobs still running; run watch() again)")
    return dash
