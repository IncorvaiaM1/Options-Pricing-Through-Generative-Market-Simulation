"""Keep every full run: logs, loss curves, scores, tables and figures, in archive/<tag>/, never overwritten.

The live folders (logs/, runs/, results/, figs/) get overwritten by the next run, so a full run is snapshotted here:

    python -m bench.archive start   --tag full1 --data bench_data/A [--models "bsm garch ..."]
    ... run models (scripts/submit_all.sh, scripts/run_model.sh or the notebook; all of them honor RUN_TAG) ...
    python -m bench.archive collect --tag full1 --data bench_data/A

scripts/submit_all.sh calls `start` for you and the report job calls `collect` (also when the report itself fails),
so normally you only set RUN_TAG:   RUN_TAG=full1 bash scripts/submit_all.sh bench_data/A

Only files written AFTER `start` for that data folder are archived, so a model that failed this time cannot sneak in
with last week's numbers; it shows up as failed or missing instead. The aggregate tables and figures are rebuilt from
those fresh files only. Rerunning one model later with the same tag adds it to the same archive (start time is kept).

archive/<tag>/
  README.md                       index of the data folders in this run
  run_info.json, git.diff, pip_freeze.txt
  <data name>/
    README.md                     status of every model, leaderboard, every figure inline
    data_config.json
    logs/<model>.log              full stdout/stderr of each model (scripts/run_model.sh writes these)
    logs/slurm/*.out              raw sbatch logs, report job included
    loss/<model>.json             train/val curves parsed from the log
    runs/<model>/*.json           meta.json (label, params, train/gen seconds), train_log.json
    runs/<model>/samples.npy      only with KEEP_SAMPLES=1 (tens of MB per model)
    results/{trackA,backtest,market}/*.json
    figs/                         paths_<label>.png per model, loss_<model>.png, loss_all.png, tables, comparison figures
"""
import os as _os
import sys as _sys

if __package__ in (None, ""):  # also allow `python path/to/file.py`
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import datetime as dt
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACK_A = ["bsm", "bootstrap", "garch", "ar_diffusion", "ar_token", "rnade", "gan", "sde_gan",
           "pure_diffusion", "path_flow", "path_flow_t"]


def _name(data):
    return os.path.basename(os.path.normpath(data))


def _sh(cmd):
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, shell=isinstance(cmd, str)).stdout
    except Exception:
        return ""


def _info_path(tag):
    return os.path.join(ROOT, "archive", tag, "run_info.json")


def _load_info(tag):
    p = _info_path(tag)
    return json.load(open(p)) if os.path.exists(p) else dict(tag=tag, data={})


def _save_info(tag, info):
    os.makedirs(os.path.dirname(_info_path(tag)), exist_ok=True)
    json.dump(info, open(_info_path(tag), "w"), indent=1)


# ------------------------------------------------------------------------------------------------ start
def preflight(data):
    """Warnings worth seeing before burning GPU hours (never blocks the run)."""
    warn = []
    cfgp = os.path.join(ROOT, data, "config.json")
    if not os.path.exists(cfgp):
        return [f"{data}/config.json not found (export the data first)"], {}
    cfg = json.load(open(cfgp))
    if cfg.get("kind", "synthetic") == "synthetic":
        n = cfg.get("n_train")
        if n is not None and n < 200000:
            warn.append(f"n_train = {n}: AR+diffusion only beat the token head at 200k windows (results/v1); "
                        f"export with --n_train 200000 for the main table")
        h = cfg.get("n_hedge_paths")
        if h is not None and h < 10000:
            warn.append(f"n_hedge_paths = {h}: README says use 10000 for final hedging numbers")
    try:
        import fatnarwhal_quant  # noqa: F401
    except Exception:
        warn.append("fatnarwhal_quant not importable: hedging columns will be empty (pip install -e <fatnarwhal-quant>)")
    return warn, cfg


def start(a):
    info = _load_info(a.tag)
    info.setdefault("created", dt.datetime.now().isoformat(timespec="seconds"))
    info["git"] = _sh(["git", "rev-parse", "HEAD"]).strip()
    info["git_dirty"] = bool(_sh(["git", "status", "--porcelain", "--untracked-files=no"]).strip())
    info["host"] = platform.node()
    info["python"] = sys.version.split()[0]
    d = os.path.join(ROOT, "archive", a.tag)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "git.diff"), "w").write(_sh(["git", "diff", "HEAD"]))
    open(os.path.join(d, "pip_freeze.txt"), "w").write(_sh([sys.executable, "-m", "pip", "freeze"]))
    for data in a.data:
        name = _name(data)
        warn, cfg = preflight(data)
        ent = info["data"].get(name)
        if ent is None or a.reset:
            ent = dict(data=data, started_at=time.time(), started=dt.datetime.now().isoformat(timespec="seconds"), models=[])
        ent["models"] = sorted(set(ent.get("models", [])) | set((a.models or "").split()), key=_order)
        ent["warnings"] = warn
        ent["config"] = cfg
        info["data"][name] = ent
        print(f"archive/{a.tag}/{name}: run clock started {ent['started']}")
        for w in warn:
            print(f"  WARNING: {w}")
    _save_info(a.tag, info)


def _order(m):
    return (TRACK_A.index(m) if m in TRACK_A else len(TRACK_A), m)


# ------------------------------------------------------------------------------------------------ collect
def _fresh(paths, t0):
    return [p for p in paths if os.path.isfile(p) and os.path.getmtime(p) >= t0 - 1]


def _copy(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
    return os.path.join(dst_dir, os.path.basename(src))


def _log_status(path):
    """(status, seconds, last error line) from a run_model.sh log."""
    txt = open(path, errors="replace").read()
    m = re.search(r"^# end .* exit (\d+) elapsed (\d+)s", txt, re.M)
    err = ""
    for line in reversed(txt.splitlines()):
        if re.search(r"(Error|Exception|Traceback|error:|Killed|CANCELLED|out of memory)", line):
            err = line.strip()[:160]
            break
    if m:
        return ("ok" if m.group(1) == "0" else f"FAILED (exit {m.group(1)})"), int(m.group(2)), err
    return ("no end marker (killed, timed out, or still running)"), None, err


def _run_py(args, out=None):
    r = subprocess.run([sys.executable, "-m"] + args, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ({' '.join(args[:2])} failed: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode})")
    if out is not None and r.returncode == 0:
        open(out, "w").write(r.stdout)
    return r.returncode == 0


def collect(a):
    info = _load_info(a.tag)
    for data in a.data:
        _collect_one(a, info, data)
    _write_top_readme(a.tag, info)


def _collect_one(a, info, data):
    name = _name(data)
    ent = info["data"].get(name)
    if ent is None:
        print(f"no `start` recorded for {name} in archive/{a.tag}; archiving files from the last 24 h")
        ent = dict(data=data, started_at=time.time() - 86400, started="(unknown, last 24 h)", models=[])
        info["data"][name] = ent
        _save_info(a.tag, info)
    t0 = ent["started_at"]
    dest = os.path.join(ROOT, "archive", a.tag, name)
    for sub in ("logs", "loss", "results", "figs"):  # rebuilt every collect, so it always reflects the live state
        shutil.rmtree(os.path.join(dest, sub), ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    cfgp = os.path.join(ROOT, data, "config.json")
    if os.path.exists(cfgp):
        shutil.copy2(cfgp, os.path.join(dest, "data_config.json"))
    kind = json.load(open(cfgp)).get("kind", "synthetic") if os.path.exists(cfgp) else "synthetic"

    # logs: one per model from run_model.sh, plus the raw slurm logs
    logs = {}
    for p in _fresh(glob.glob(os.path.join(ROOT, "logs", "models", name, "*.log")), t0):
        logs[os.path.splitext(os.path.basename(p))[0]] = _copy(p, os.path.join(dest, "logs"))
    for p in _fresh(glob.glob(os.path.join(ROOT, "logs", f"*_{name}-*.out")), t0):
        _copy(p, os.path.join(dest, "logs", "slurm"))
    stale_logs = sorted(set(os.path.splitext(os.path.basename(p))[0]
                            for p in glob.glob(os.path.join(ROOT, "logs", "models", name, "*.log"))) - set(logs))

    # run metadata (+ samples on request)
    metas, fresh_samples = {}, []
    for mdir in sorted(glob.glob(os.path.join(ROOT, "runs", name, "*"))):
        m = os.path.basename(mdir)
        for p in _fresh(glob.glob(os.path.join(mdir, "*.json")), t0):
            q = _copy(p, os.path.join(dest, "runs", m))
            if os.path.basename(p) == "meta.json":
                metas[m] = json.load(open(q))
        s = os.path.join(mdir, "samples.npy")
        if _fresh([s], t0):
            fresh_samples.append(s)
            if os.environ.get("KEEP_SAMPLES") == "1":
                _copy(s, os.path.join(dest, "runs", m))

    # scores
    res = {}
    for kindir in ("trackA", "backtest", "market"):
        fs = _fresh(glob.glob(os.path.join(ROOT, "results", name, kindir, "*.json")), t0)
        res[kindir] = [_copy(p, os.path.join(dest, "results", kindir)) for p in fs]

    # per-model figures written during the run (paths_<label>.png)
    figs = os.path.join(dest, "figs")
    os.makedirs(figs, exist_ok=True)
    for p in _fresh(glob.glob(os.path.join(ROOT, "figs", name, "paths_*.png")), t0):
        _copy(p, figs)

    # loss curves from the logs
    try:
        from bench.nb import loss_figure, parse_log
        hists = {}
        for m in sorted(logs, key=_order):
            h = parse_log(logs[m])
            if h:
                hists[m] = h
                os.makedirs(os.path.join(dest, "loss"), exist_ok=True)
                json.dump(h, open(os.path.join(dest, "loss", f"{m}.json"), "w"))
                png = loss_figure({m: h}, f"{m} on {name}")
                if png:
                    open(os.path.join(figs, f"loss_{m}.png"), "wb").write(png)
        png = loss_figure(hists, f"training curves, {name}, run {a.tag}")
        if png:
            open(os.path.join(figs, "loss_all.png"), "wb").write(png)
    except Exception as e:  # never lose the archive over a plotting problem
        print(f"  (loss curves skipped: {e})")

    # aggregate tables + figures, from this run's files only
    table_md = None
    if kind == "real":
        if res["backtest"]:
            table_md = os.path.join(figs, "backtest_table.md")
            _run_py(["bench.backtest", "table"] + res["backtest"], table_md)
            _run_py(["bench.plots", "backtest", "--results"] + res["backtest"] + ["--out", figs])
        if res["market"]:
            _run_py(["bench.plots", "market", "--results"] + res["market"] + ["--out", figs])
    elif res["trackA"]:
        table_md = os.path.join(figs, "trackA_table.md")
        _run_py(["bench.trackA", "table"] + res["trackA"], table_md)
        _run_py(["bench.plots", "leaderboard", "--results"] + res["trackA"] + ["--out", figs])
        if fresh_samples:
            _run_py(["bench.plots", "smile", "--data", data, "--samples"] + fresh_samples + ["--out", figs])
    if fresh_samples:
        _run_py(["bench.plots", "terminal", "--data", data, "--samples"] + fresh_samples + ["--out", figs])

    # status of every model
    expected = sorted(set(ent.get("models") or []) | set(logs) | set(metas), key=_order)
    rows = []
    for m in expected:
        if m in logs:
            st, sec, err = _log_status(logs[m])
        else:
            st, sec, err = ("missing (no log after start)" + (", only an older log exists" if m in stale_logs else "")), None, ""
        meta = metas.get(m, {})
        rows.append(dict(model=m, status=st, minutes=None if sec is None else round(sec / 60, 1), label=meta.get("label", ""),
                         n_params=meta.get("n_params"), train_s=meta.get("train_seconds"), gen_s=meta.get("gen_seconds"),
                         error=err if not st.startswith("ok") else ""))
    json.dump(rows, open(os.path.join(dest, "status.json"), "w"), indent=1)
    _write_readme(a.tag, ent, name, dest, rows, table_md, kind)
    n_ok = sum(r["status"] == "ok" for r in rows)
    print(f"archive/{a.tag}/{name}: {n_ok}/{len(rows)} models ok, {sum(len(v) for v in res.values())} score files, "
          f"{len(glob.glob(os.path.join(figs, '*.png')))} figures")
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['model']}: {r['status']}  {r['error']}")


def _fmt(x, nd=0):
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return f"{x:,.{nd}f}"
    return str(x)


def _write_readme(tag, ent, name, dest, rows, table_md, kind):
    L = [f"# {name}, run `{tag}`", "",
         f"Started {ent.get('started')}, data `{ent.get('data')}` ({kind}). Collected "
         f"{dt.datetime.now().isoformat(timespec='seconds')}. Only files written after the start are included.", ""]
    for w in ent.get("warnings", []):
        L.append(f"> **Warning:** {w}")
    if ent.get("warnings"):
        L.append("")
    L += ["## Models", "", "| model | status | min | label | params | train s | gen s | log | last error |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        log = f"[log](logs/{r['model']}.log)" if os.path.exists(os.path.join(dest, "logs", f"{r['model']}.log")) else ""
        L.append(f"| {r['model']} | {r['status']} | {_fmt(r['minutes'], 1)} | {r['label']} | {_fmt(r['n_params'])} | "
                 f"{_fmt(r['train_s'])} | {_fmt(r['gen_s'], 1)} | {log} | {r['error'].replace('|', '/')} |")
    L.append("")
    if table_md and os.path.exists(table_md):
        L += ["## Leaderboard", "", open(table_md).read().strip(), ""]
    figs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dest, "figs", "*.png")))
    groups = [("Comparison", [f for f in figs if not f.startswith(("paths_", "loss_"))]),
              ("Training curves", [f for f in figs if f == "loss_all.png"] + [f for f in figs if f.startswith("loss_") and f != "loss_all.png"]),
              ("Simulated paths", [f for f in figs if f.startswith("paths_")])]
    for title, fs in groups:
        if fs:
            L += [f"## {title}", ""] + [f"**{f[:-4]}**\n\n![{f}](figs/{f})\n" for f in fs]
    open(os.path.join(dest, "README.md"), "w").write("\n".join(L) + "\n")


def _write_top_readme(tag, info):
    d = os.path.join(ROOT, "archive", tag)
    L = [f"# Run `{tag}`", "", f"git `{info.get('git', '')[:10]}`" + (" (uncommitted changes, see git.diff)" if info.get("git_dirty") else "")
         + f", host `{info.get('host', '')}`, python {info.get('python', '')}", ""]
    for name, ent in info["data"].items():
        sp = os.path.join(d, name, "status.json")
        ok = ""
        if os.path.exists(sp):
            rows = json.load(open(sp))
            ok = f": {sum(r['status'] == 'ok' for r in rows)}/{len(rows)} models ok"
        L.append(f"- [{name}]({name}/README.md), started {ent.get('started')}{ok}")
    open(os.path.join(d, "README.md"), "w").write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("start", "collect"):
        p = sub.add_parser(c)
        p.add_argument("--tag", default=os.environ.get("RUN_TAG"))
        p.add_argument("--data", nargs="+", required=True)
    sub.choices["start"].add_argument("--models", default=os.environ.get("MODELS", ""))
    sub.choices["start"].add_argument("--reset", action="store_true", help="restart the clock (ignore earlier files)")
    a = ap.parse_args(argv)
    if not a.tag:
        ap.error("give --tag or set RUN_TAG")
    {"start": start, "collect": collect}[a.cmd](a)


if __name__ == "__main__":
    main()
