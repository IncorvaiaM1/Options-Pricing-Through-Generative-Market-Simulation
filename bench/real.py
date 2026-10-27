"""Real data (yfinance) for Track A, in the same file format as the synthetic bench_data/A.

Step 1, on a machine with internet (PACE login node, laptop):
    python -m bench.real download --preset megacap_tech --start 2010-01-01 --out bench_data/real_raw/megacap_tech
  writes prices.csv (adjusted close, one column per ticker) and rates.csv (13-week T-bill, ^IRX, annualized).
  Any CSV with a Date column plus one column of prices per ticker works too (--prices_csv), e.g. for offline use.

Step 2, anywhere:
    python -m bench.real export --raw bench_data/real_raw/megacap_tech --out bench_data/real/megacap_tech \
        --train_end 2021-12-31 --val_end 2023-06-30
  Time-based split (no look-ahead): train windows end before train_end, validation windows before val_end,
  evaluation contexts start after val_end. Writes train.npy / val.npy / eval_ctx.npy like bench/A, plus
  eval_realized.npy [E, H] (what actually happened next), eval_meta.json (ticker, date, spot), eval_r.npy (rate).

A single ticker is a one-item list; a preset of similar stocks gives the pitch's "clusters of similar stocks"
(windows from all tickers are pooled; returns are in raw daily log-return units, and the models see each
window's own volatility level through the context).
"""
import os as _os
import sys as _sys

if __package__ in (None, ""):  # also allow `python path/to/file.py`, not only `python -m pkg.file`
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import warnings

warnings.filterwarnings("ignore", message=".*inplace keyword.*")  # yfinance + pandas 3 deprecation noise
import pandas as pd

PRESETS = {
    "spy": ["SPY"],
    "index_etfs": ["SPY", "QQQ", "IWM", "DIA"],
    "megacap_tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    "banks": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "staples": ["PG", "KO", "PEP", "WMT", "COST"],
}


def download(args):
    import yfinance as yf
    tickers = args.tickers or PRESETS[args.preset]
    os.makedirs(args.out, exist_ok=True)
    df = yf.download(tickers, start=args.start, end=args.end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    df = df.dropna(how="all")
    if df.empty:
        raise SystemExit("yfinance returned no data (no internet on this node? run this step on a login node).")
    df.index.name = "Date"
    df.to_csv(os.path.join(args.out, "prices.csv"))
    try:
        irx = yf.download("^IRX", start=args.start, end=args.end, auto_adjust=False, progress=False)["Close"]
        irx = (irx.squeeze() / 100.0).rename("r").to_frame()
        irx.index.name = "Date"
        irx.to_csv(os.path.join(args.out, "rates.csv"))
    except Exception as e:  # rates are optional; export falls back to a constant
        print("could not download ^IRX rates:", e)
    print(f"saved {df.shape[1]} tickers x {df.shape[0]} days to {args.out}: {list(df.columns)}")


def windows(lr, L, H, stride):
    """lr: 1-D array of log returns -> [n, L+H] windows."""
    n = (len(lr) - (L + H)) // stride + 1
    if n <= 0:
        return np.empty((0, L + H))
    idx = np.arange(n)[:, None] * stride + np.arange(L + H)[None, :]
    return lr[idx]


def export(args):
    raw = args.raw
    px = pd.read_csv(args.prices_csv or os.path.join(raw, "prices.csv"), index_col="Date", parse_dates=True).sort_index()
    rates_path = os.path.join(raw, "rates.csv") if raw else None
    if rates_path and os.path.exists(rates_path):
        r = pd.read_csv(rates_path, index_col="Date", parse_dates=True)["r"].reindex(px.index).ffill().bfill()
    else:
        r = pd.Series(args.const_rate, index=px.index)
        print(f"no rates.csv; using constant r = {args.const_rate}")
    L, H = args.ctx_len, args.horizon
    te, ve = pd.Timestamp(args.train_end), pd.Timestamp(args.val_end)
    TR, VA, CTX, REAL, META, RR = [], [], [], [], [], []
    for tk in px.columns:
        s = px[tk].dropna()
        s = s[s > 0]
        lr = np.log(s).diff().dropna()
        dates = lr.index
        v = lr.values
        # clip absurd prints (bad ticks surviving adjustment): |log return| <= 0.7, i.e. about -50% / +100% in a day
        v = np.clip(v, -0.7, 0.7)
        tr_mask = dates <= te
        va_mask = (dates > te) & (dates <= ve)
        TR.append(windows(v[tr_mask], L, H, args.train_stride))
        VA.append(windows(v[va_mask], L, H, args.train_stride))
        # evaluation: contexts that END after val_end, every eval_stride days, with H realized days after
        start = np.searchsorted(dates, ve, side="right")
        for end in range(max(start, L), len(v) - H + 1, args.eval_stride):
            CTX.append(v[end - L:end]); REAL.append(v[end:end + H])
            d = dates[end - 1]
            META.append(dict(ticker=str(tk), date=str(d.date()), spot=float(s.loc[d])))
            RR.append(float(r.loc[d]))
    train = np.concatenate(TR); val = np.concatenate(VA)
    if len(train) == 0 or len(CTX) == 0:
        raise SystemExit(f"nothing to export: {len(train)} training windows, {len(CTX)} eval contexts. "
                         f"Check the date range of the prices vs --train_end/--val_end, and for NaN prices.")
    ctx = np.array(CTX); real = np.array(REAL)
    if args.max_eval and len(ctx) > args.max_eval:
        keep = np.sort(np.random.default_rng(0).choice(len(ctx), args.max_eval, replace=False))
        ctx, real = ctx[keep], real[keep]
        META = [META[i] for i in keep]; RR = [RR[i] for i in keep]
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "train.npy"), train.astype(np.float32))
    np.save(os.path.join(args.out, "val.npy"), val.astype(np.float32))
    np.save(os.path.join(args.out, "eval_ctx.npy"), ctx.astype(np.float32))
    np.save(os.path.join(args.out, "eval_realized.npy"), real.astype(np.float32))
    np.save(os.path.join(args.out, "eval_r.npy"), np.array(RR))
    json.dump(META, open(os.path.join(args.out, "eval_meta.json"), "w"))
    cfg = dict(kind="real", tickers=[str(t) for t in px.columns], ctx_len=L, horizon=H, n_train=int(len(train)),
               n_val=int(len(val)), n_eval=int(len(ctx)), scale=float(train.std()), train_end=args.train_end,
               val_end=args.val_end, ks=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], r_mean=float(np.mean(RR)) if RR else 0.0,
               nu=None, data_seed=None)
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    print(f"exported {args.out}: train {train.shape}, val {val.shape}, eval {ctx.shape} "
          f"({len(set(m['ticker'] for m in META))} tickers, {META[0]['date'] if META else '-'} .. {META[-1]['date'] if META else '-'})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--preset", default="spy", choices=list(PRESETS)); d.add_argument("--tickers", nargs="*")
    d.add_argument("--start", default="2010-01-01"); d.add_argument("--end", default=None)
    d.add_argument("--out", required=True)
    e = sub.add_parser("export")
    e.add_argument("--raw", default=None); e.add_argument("--prices_csv", default=None)
    e.add_argument("--out", required=True)
    e.add_argument("--train_end", default="2021-12-31"); e.add_argument("--val_end", default="2023-06-30")
    e.add_argument("--ctx_len", type=int, default=64); e.add_argument("--horizon", type=int, default=20)
    e.add_argument("--train_stride", type=int, default=1); e.add_argument("--eval_stride", type=int, default=5)
    e.add_argument("--max_eval", type=int, default=0)
    e.add_argument("--const_rate", type=float, default=0.03)
    args = ap.parse_args(argv)
    download(args) if args.cmd == "download" else export(args)


if __name__ == "__main__":
    main()
