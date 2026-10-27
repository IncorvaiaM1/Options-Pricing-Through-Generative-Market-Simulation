"""Compare model prices with CURRENT market option prices (the pitch's "current marketplace values").

yfinance only serves today's option chains (no history), so this is a snapshot comparison.

1) snapshot (needs internet; PACE login node or laptop). Uses a real export for the model settings (L, H, scale):
     python -m bench.market snapshot --export bench_data/real/megacap_tech --out bench_data/market/2026-10-27
   For each ticker: the last L daily log returns (the context), and the option expiry whose number of trading
   days to expiry is closest to H, with OTM puts/calls (bid, ask, mid, volume, open interest), spot and the 13-week
   T-bill rate. The folder is a valid --data / --also_eval target: eval_ctx.npy has one row per ticker.

2) generate samples for it: any starter with  --also_eval bench_data/market/2026-10-27  (writes
   runs/<model>/also_2026-10-27/samples.npy), or the classical baselines directly:
     python -m starters.a0_baselines --method bsm --data bench_data/market/2026-10-27 --out runs/mkt_bsm

3) compare:
     python -m bench.market compare --market bench_data/market/2026-10-27 --samples runs/a6/also_2026-10-27/samples.npy
   Model prices are risk-neutral adjusted (paths rescaled so E[S_T] = forward) and compared in implied-vol
   points with the market mid. The unadjusted (real-world) version is also reported; its gap to the market is
   mostly the volatility risk premium, not model error.
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import shutil

import numpy as np

from bench.finance import implied_vol


def snapshot(args):
    import yfinance as yf
    ecfg = json.load(open(os.path.join(args.export, "config.json")))
    L, H = ecfg["ctx_len"], ecfg["horizon"]
    tickers = args.tickers or ecfg["tickers"]
    today = dt.date.today()
    try:
        r = float(yf.Ticker("^IRX").history(period="5d")["Close"].dropna().iloc[-1]) / 100
    except Exception:
        r = args.const_rate
        print(f"could not get ^IRX; using r = {r}")
    ctxs, quotes = [], []
    for tk in tickers:
        t = yf.Ticker(tk)
        hist = t.history(period="1y", auto_adjust=True)["Close"].dropna()
        lr = np.diff(np.log(hist.values))
        if len(lr) < L or not t.options:
            print(f"skip {tk}: not enough history or no options"); continue
        exps = [(e, int(np.busday_count(today, dt.date.fromisoformat(e)))) for e in t.options]
        exps = [e for e in exps if e[1] >= 5]
        exp, td = min(exps, key=lambda e: abs(e[1] - H))
        ch = t.option_chain(exp)
        spot = float(hist.iloc[-1])
        rows = []
        for kind, df in (("put", ch.puts), ("call", ch.calls)):
            for _, q in df.iterrows():
                K = float(q["strike"])
                if (kind == "put" and K >= spot) or (kind == "call" and K < spot):
                    continue  # keep the OTM side only
                bid, ask = float(q.get("bid", 0) or 0), float(q.get("ask", 0) or 0)
                rows.append(dict(kind=kind, strike=K, bid=bid, ask=ask, mid=0.5 * (bid + ask), last=float(q.get("lastPrice", 0) or 0),
                                 volume=float(q.get("volume", 0) or 0), open_interest=float(q.get("openInterest", 0) or 0),
                                 yahoo_iv=float(q.get("impliedVolatility", float("nan")))))
        ctxs.append(lr[-L:]); quotes.append(dict(ticker=tk, spot=spot, expiry=exp, trading_days=td, r=r, options=rows))
        print(f"{tk}: spot {spot:.2f}, expiry {exp} ({td} trading days), {len(rows)} OTM quotes")
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "eval_ctx.npy"), np.array(ctxs, np.float32))
    json.dump(quotes, open(os.path.join(args.out, "quotes.json"), "w"), indent=1)
    cfg = dict(ecfg, kind="market", n_eval=len(ctxs), snapshot_date=str(today), source_export=os.path.abspath(args.export),
               tickers=[q["ticker"] for q in quotes])
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    for f in ("train.npy", "val.npy"):  # so baselines that fit on training data (GARCH) work from this folder
        src = os.path.join(args.export, f)
        if os.path.exists(src) and not os.path.exists(os.path.join(args.out, f)):
            try:
                os.symlink(os.path.abspath(src), os.path.join(args.out, f))
            except OSError:
                shutil.copy(src, os.path.join(args.out, f))
    print(f"saved snapshot for {len(ctxs)} tickers to {args.out}")


def compare(args):
    quotes = json.load(open(os.path.join(args.market, "quotes.json")))
    ctx = np.load(os.path.join(args.market, "eval_ctx.npy")).astype(np.float64)
    smp = np.load(args.samples).astype(np.float64)
    assert smp.shape[0] == len(quotes), "samples must be generated for this snapshot's eval_ctx.npy"
    mp = os.path.join(os.path.dirname(args.samples), "meta.json")
    label = args.label or (json.load(open(mp)).get("label") if os.path.exists(mp) else None) or os.path.basename(os.path.dirname(args.samples))
    out = dict(label=label, tickers={})
    allerr, allerr_P, allerr_bsm = [], [], []
    for i, q in enumerate(quotes):
        H = smp.shape[2]
        h = min(q["trading_days"], H)
        if q["trading_days"] > H:
            print(f"  note {q['ticker']}: expiry is {q['trading_days']} trading days out, model horizon {H}; using {H}")
        T = q["trading_days"] / 252
        r, S0 = q["r"], q["spot"]
        F = S0 * math.exp(r * T)
        G = np.exp(smp[i, :, :h].sum(-1))
        Gq = G * math.exp(r * T) / G.mean()
        sig = math.sqrt(np.mean(ctx[i] ** 2) * 252)
        rows = [o for o in q["options"] if o["bid"] > 0 and o["ask"] > o["bid"] and (o["ask"] - o["bid"]) / o["mid"] < args.max_rel_spread]
        rows = [o for o in rows if abs(math.log(o["strike"] / F)) <= args.max_std * sig * math.sqrt(T)]
        if not rows:
            continue
        K = np.array([o["strike"] for o in rows]); put = np.array([o["kind"] == "put" for o in rows])
        x = np.log(K / F)
        mkt = np.array([o["mid"] for o in rows]) * math.exp(r * T) / F           # undiscounted, forward-normalized
        def model_price(Gx):
            S = S0 * Gx / F
            k = K / F
            return np.where(put[:, None], np.maximum(k[:, None] - S[None], 0), np.maximum(S[None] - k[:, None], 0)).mean(1)
        iv_mkt = implied_vol(mkt, x, T)
        iv_Q = implied_vol(model_price(Gq), x, T)
        iv_P = implied_vol(model_price(G * math.exp(r * T)), x, T)  # real-world paths, discounted at r
        e, eP, eb = np.abs(iv_Q - iv_mkt) * 100, np.abs(iv_P - iv_mkt) * 100, np.abs(sig - iv_mkt) * 100
        allerr += list(e[np.isfinite(e)]); allerr_P += list(eP[np.isfinite(eP)]); allerr_bsm += list(eb[np.isfinite(eb)])
        out["tickers"][q["ticker"]] = dict(expiry=q["expiry"], trading_days=q["trading_days"], ctx_vol=sig,
                                           log_moneyness=x.tolist(), iv_market=iv_mkt.tolist(), iv_model_Q=iv_Q.tolist(),
                                           iv_model_P=iv_P.tolist(), mean_abs_iv_err_Q=float(np.nanmean(e)))
        print(f"  {q['ticker']}: {len(rows)} quotes; |IV err| model {np.nanmean(e):.2f}  (real-world paths {np.nanmean(eP):.2f}, "
              f"BSM at ctx vol {np.nanmean(eb):.2f}) vol pts")
    out["mean_abs_iv_err_Q"] = float(np.mean(allerr)) if allerr else None
    out["mean_abs_iv_err_P"] = float(np.mean(allerr_P)) if allerr_P else None
    out["mean_abs_iv_err_bsm_ctxvol"] = float(np.mean(allerr_bsm)) if allerr_bsm else None
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, f"{label}.json"), "w"), indent=1)
    print(f"[{label}] mean |IV error| vs market: {out['mean_abs_iv_err_Q']}  (real-world: {out['mean_abs_iv_err_P']}, "
          f"BSM at ctx vol: {out['mean_abs_iv_err_bsm_ctxvol']})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--export", required=True); s.add_argument("--out", required=True)
    s.add_argument("--tickers", nargs="*"); s.add_argument("--const_rate", type=float, default=0.03)
    c = sub.add_parser("compare")
    c.add_argument("--market", required=True); c.add_argument("--samples", required=True)
    c.add_argument("--label"); c.add_argument("--out", default="results/market")
    c.add_argument("--max_rel_spread", type=float, default=0.5); c.add_argument("--max_std", type=float, default=2.5)
    a = ap.parse_args(argv)
    snapshot(a) if a.cmd == "snapshot" else compare(a)


if __name__ == "__main__":
    main()
