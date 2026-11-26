"""Historical option prices from Massive (formerly Polygon.io) for the real-data market comparison.

The yfinance snapshot (bench/market.py snapshot) gives one date. This builds the same kind of market folder for every
evaluation date of a real export that Massive has data for, so `bench.market compare` scores models against the market
over many dates instead of one.

Data: Massive Options Starter includes the day-aggregates flat files (us_options_opra/day_aggs_v1): one gzipped CSV per
trading day with open/high/low/close/volume for EVERY US option contract, going back 2 years on Starter. There is no
bid/ask on Starter, so the market price is the day's last trade (close). bench.market compare therefore filters these on
volume and a minimum price instead of on the spread, and the spot is backed out of the options by put-call parity
(the real export uses dividend-adjusted closes, which are off from the traded price by up to about a percent).

1) download (needs internet: PACE login node or laptop), either way; both write the same files:
   a) flat files (fast: one file per date), with the S3 keys from the Massive dashboard (not the API key):
        export MASSIVE_S3_ACCESS_KEY=...   MASSIVE_S3_SECRET_KEY=...
   b) REST, with the plain API key (slower: one request per option contract, cached so reruns resume):
        export MASSIVE_API_KEY=...
   then
        python -m bench.massive download --export bench_data/real/megacap_tech --out bench_data/massive_raw
   It uses the flat files when the S3 keys are set and REST otherwise (force with --source rest|s3). Keys go in
   ~/.bashrc, never in the repo. Dates already on disk are skipped, so it is safe to rerun.

2) build (anywhere, no internet):
     python -m bench.massive build --export bench_data/real/megacap_tech --raw bench_data/massive_raw \
         --out bench_data/market/massive_megacap_tech
   For each (ticker, date) context: the expiry with the most trading days <= the model horizon, its OTM puts and
   calls that traded that day, and the parity spot. The folder works as MARKET=... for run_model.sh / submit_all.sh.

3) MARKET=bench_data/market/massive_megacap_tech bash scripts/run_model.sh <model> bench_data/real/megacap_tech
"""
import os as _os
import sys as _sys

if __package__ in (None, ""):  # also allow `python path/to/file.py`
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import datetime as dt
import gzip
import io
import json
import math
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ENDPOINT = "https://files.massive.com"
BUCKET = "flatfiles"
PREFIX = "us_options_opra/day_aggs_v1"
_REST = re.compile(r"^(\d{6})([CP])(\d{8})$")  # after "O:<root>": YYMMDD, C/P, strike * 1000


def _client():
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise SystemExit("needs boto3:  pip install boto3")
    key, sec = os.environ.get("MASSIVE_S3_ACCESS_KEY"), os.environ.get("MASSIVE_S3_SECRET_KEY")
    if not key or not sec:
        raise SystemExit("set MASSIVE_S3_ACCESS_KEY and MASSIVE_S3_SECRET_KEY (Massive dashboard, Flat Files / S3 keys; "
                         "not the REST API key). Put them in ~/.bashrc, never in the repo.")
    return boto3.client("s3", endpoint_url=ENDPOINT, aws_access_key_id=key, aws_secret_access_key=sec,
                        config=Config(signature_version="s3v4", retries=dict(max_attempts=5)))


def _filter(df, tickers):
    """Keep standard contracts on these underlyings (drops adjusted roots like AAPL1 and look-alikes like GOOG vs GOOGL)."""
    keep = np.zeros(len(df), bool)
    t = df["ticker"].astype(str)
    for tk in tickers:
        p = "O:" + tk
        m = t.str.startswith(p).values
        if m.any():
            rest = t[m].str.slice(len(p))
            keep[np.where(m)[0][rest.str.match(_REST).values]] = True
    return df[keep]


def download_s3(a, dates, tickers):
    todo = [d for d in dates if not os.path.exists(os.path.join(a.out, f"{d}.csv.gz"))]
    if not todo:
        return
    s3 = _client()
    t0, n_ok, denied, missing = time.time(), 0, 0, []
    for i, d in enumerate(todo):
        y, m = d[:4], d[5:7]
        key = f"{PREFIX}/{y}/{m}/{d}.csv.gz"
        try:
            body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        except Exception as e:  # botocore ClientError: NoSuchKey (no file that day) or 403 (outside the plan's history)
            code = getattr(e, "response", {}).get("Error", {}).get("Code", type(e).__name__)
            if code in ("403", "AccessDenied", "Forbidden"):
                denied += 1
                print(f"  {d}: access denied (outside the plan's 2-year window, or wrong keys)")
                if denied >= 3 and n_ok == 0:
                    raise SystemExit("3 denials and no successes: check the S3 keys (dashboard > Flat Files), not the API key")
            else:
                missing.append(d)
                print(f"  {d}: {code}")
            continue
        df = pd.read_csv(io.BytesIO(body), compression="gzip")
        sub = _filter(df, tickers)
        tmp = os.path.join(a.out, f".{d}.csv.gz")
        sub.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, os.path.join(a.out, f"{d}.csv.gz"))  # atomic: a killed run never leaves a half file
        n_ok += 1
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"  {i + 1}/{len(todo)}  {d}: {len(sub)} contracts kept of {len(df)}  {time.time() - t0:.0f}s", flush=True)
    print(f"downloaded {n_ok} days to {a.out}" + (f"; {denied} denied" if denied else "") + (f"; missing {missing}" if missing else ""))


def download(a):
    meta = json.load(open(os.path.join(a.export, "eval_meta.json")))
    tickers = sorted(set(m["ticker"] for m in meta))
    today = dt.date.today()
    since = dt.date.fromisoformat(a.since) if a.since else today - dt.timedelta(days=730 - 7)  # Starter: 2 years back
    dates = sorted(set(m["date"] for m in meta))
    dates = [d for d in dates if since <= dt.date.fromisoformat(d) < today]
    os.makedirs(a.out, exist_ok=True)
    done = sum(os.path.exists(os.path.join(a.out, f"{d}.csv.gz")) for d in dates)
    src = a.source
    if src == "auto":
        src = "s3" if os.environ.get("MASSIVE_S3_ACCESS_KEY") and os.environ.get("MASSIVE_S3_SECRET_KEY") else "rest"
    print(f"{len(dates)} evaluation dates since {since} for {tickers}; {done} already downloaded; source: {src}")
    if done == len(dates):
        return
    if src == "s3":
        download_s3(a, dates, tickers)
    else:
        download_rest(a, meta, dates, tickers)


# ------------------------------------------------------------------------------------------------ REST
API = "https://api.massive.com"


def _get(session, url, params=None, tries=6):
    """GET with retries on timeouts, connection errors, 429 and 5xx (exponential backoff)."""
    import requests
    for i in range(tries):
        try:
            r = session.get(url, params=params, timeout=60)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if i == tries - 1:
                raise
            time.sleep(2 ** i); continue
        if r.status_code == 429 or r.status_code >= 500:
            if i == tries - 1:
                r.raise_for_status()
            time.sleep(min(60, 2 ** i)); continue
        if r.status_code in (401, 403):
            raise SystemExit(f"Massive refused the request ({r.status_code}): check MASSIVE_API_KEY and the plan's date range. "
                             f"{r.text[:200]}")
        r.raise_for_status()
        return r.json()


def _paged(session, url, params):
    out = []
    while url:
        js = _get(session, url, params)
        out += js.get("results", []) or []
        url, params = js.get("next_url"), None  # next_url already carries the query
    return out


def download_rest(a, meta, dates, tickers):
    """Same output as the flat files, from the REST API: for each (ticker, date), the contracts listed that day on the
    expiry closest to (not past) the horizon, then each contract's daily bars. Everything is cached under
    <out>/_rest_cache, so an interrupted run resumes where it stopped."""
    import requests
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise SystemExit("set MASSIVE_API_KEY (or the S3 keys for flat files); put it in ~/.bashrc, never in the repo")
    cfg = json.load(open(os.path.join(a.export, "config.json")))
    H = cfg["horizon"]
    sess = requests.Session()
    sess.headers["Authorization"] = f"Bearer {key}"
    cache = os.path.join(a.out, "_rest_cache")
    os.makedirs(os.path.join(cache, "bars"), exist_ok=True)
    spot = {(m["ticker"], m["date"]): m["spot"] for m in meta}
    want = sorted((tk, d) for (tk, d) in spot if d in set(dates)
                  and not os.path.exists(os.path.join(a.out, f"{d}.csv.gz")))

    # 1) contracts per (ticker, date): one expiry, strikes within +-band of spot, calls and puts
    need = {}  # contract -> set of dates
    t0 = time.time()
    for j, (tk, d) in enumerate(want):
        cp = os.path.join(cache, f"contracts_{tk}_{d}.json")
        if os.path.exists(cp):
            cs = json.load(open(cp))
        else:
            day = dt.date.fromisoformat(d)
            S = spot[(tk, d)]
            cs = _paged(sess, f"{API}/v3/reference/options/contracts", {
                "underlying_ticker": tk, "as_of": d, "expired": "false", "limit": 1000,
                "expiration_date.gte": str(day + dt.timedelta(days=a.min_days)),
                "expiration_date.lte": str(day + dt.timedelta(days=int(H * 7 / 5) + 4)),
                "strike_price.gte": round(S * (1 - a.strike_band), 2), "strike_price.lte": round(S * (1 + a.strike_band), 2)})
            cs = [dict(ticker=c["ticker"], expiration_date=c["expiration_date"]) for c in cs]
            json.dump(cs, open(cp, "w"))
        day = dt.date.fromisoformat(d)
        exps = {c["expiration_date"] for c in cs}
        tds = {e: int(np.busday_count(day, dt.date.fromisoformat(e))) for e in exps}
        ok = [e for e in exps if a.min_days <= tds[e] <= H]
        if ok:
            exp = max(ok, key=lambda e: tds[e])
            for c in cs:
                if c["expiration_date"] == exp:
                    need.setdefault(c["ticker"], set()).add(d)
        if (j + 1) % 25 == 0 or j + 1 == len(want):
            print(f"  contracts: {j + 1}/{len(want)} (ticker, date) pairs, {len(need)} contracts so far  {time.time() - t0:.0f}s", flush=True)

    # 2) daily bars per contract (parallel, cached)
    def bars(sym):
        bp = os.path.join(cache, "bars", sym.replace(":", "_") + ".json")
        if os.path.exists(bp):
            return sym, json.load(open(bp))
        ds = sorted(need[sym])
        res = _paged(sess, f"{API}/v2/aggs/ticker/{sym}/range/1/day/{ds[0]}/{ds[-1]}", {"adjusted": "false", "limit": 50000})
        tmp = bp + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, bp)
        return sym, res
    syms = sorted(need)
    rows, t0 = {}, time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (sym, res) in enumerate(ex.map(bars, syms)):
            for b in res:
                d = str(pd.Timestamp(b["t"], unit="ms", tz="UTC").tz_convert("America/New_York").date())
                if d in need[sym] and b.get("c") is not None:
                    rows.setdefault(d, []).append(dict(ticker=sym, volume=b.get("v", 0), open=b.get("o"), close=b["c"],
                                                       high=b.get("h"), low=b.get("l"), window_start=b["t"] * 1_000_000,
                                                       transactions=b.get("n")))
            if (i + 1) % 200 == 0 or i + 1 == len(syms):
                el = time.time() - t0
                print(f"  bars: {i + 1}/{len(syms)} contracts  {el:.0f}s (about {el / (i + 1) * (len(syms) - i - 1) / 60:.0f} min left)", flush=True)

    # 3) one file per date, same columns as the flat files
    for d in sorted(set(d for _, d in want)):
        df = pd.DataFrame(rows.get(d, []), columns=["ticker", "volume", "open", "close", "high", "low", "window_start", "transactions"])
        tmp = os.path.join(a.out, f".{d}.csv.gz")
        df.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, os.path.join(a.out, f"{d}.csv.gz"))
    print(f"wrote {len(set(d for _, d in want))} days to {a.out} ({sum(len(v) for v in rows.values())} contract-days)")


def _parse(df, tk):
    p = "O:" + tk
    df = df[df["ticker"].str.startswith(p)].copy()
    m = df["ticker"].str.slice(len(p)).str.extract(_REST)
    df["expiry"] = pd.to_datetime(m[0], format="%y%m%d").dt.date
    df["kind"] = np.where(m[1] == "C", "call", "put")
    df["strike"] = m[2].astype(float) / 1000.0
    return df.dropna(subset=["expiry"])


def _parity_forward(ch, T, r, s_ref, n_near=3):
    """Forward from put-call parity at the strikes nearest s_ref where both sides traded: F = K + e^{rT} (C - P)."""
    c = ch[ch.kind == "call"].set_index("strike")["close"]
    p = ch[ch.kind == "put"].set_index("strike")["close"]
    ks = sorted(set(c.index) & set(p.index), key=lambda k: abs(k - s_ref))[:n_near]
    if not ks:
        return None
    return float(np.median([k + math.exp(r * T) * (c[k] - p[k]) for k in ks]))


def build(a):
    cfg = json.load(open(os.path.join(a.export, "config.json")))
    L, H = cfg["ctx_len"], cfg["horizon"]
    ctx = np.load(os.path.join(a.export, "eval_ctx.npy"))
    real = np.load(os.path.join(a.export, "eval_realized.npy"))
    rr = np.load(os.path.join(a.export, "eval_r.npy"))
    meta = json.load(open(os.path.join(a.export, "eval_meta.json")))
    cache, quotes, keep, skipped = {}, [], [], {}
    n_parity = 0
    for i, mt in enumerate(meta):
        d, tk = mt["date"], mt["ticker"]
        f = os.path.join(a.raw, f"{d}.csv.gz")
        if not os.path.exists(f):
            skipped["no file for date"] = skipped.get("no file for date", 0) + 1
            continue
        if d not in cache:
            cache[d] = pd.read_csv(f)  # filtered files are tens of KB, so keeping every day in memory is fine
        df = _parse(cache[d], tk)
        df = df[(df["volume"] >= 1) & (df["close"] > 0)]
        day = dt.date.fromisoformat(d)
        tds = {e: int(np.busday_count(day, e)) for e in df["expiry"].unique()}
        # expiry with the most trading days that still fits the model horizon (megacaps list weeklies, so ~16-20 days)
        ok = [e for e, td in tds.items() if a.min_days <= td <= H and (df["expiry"] == e).sum() >= a.min_contracts]
        if not ok:
            skipped["no expiry with enough trades"] = skipped.get("no expiry with enough trades", 0) + 1
            continue
        exp = max(ok, key=lambda e: tds[e])
        ch = df[df["expiry"] == exp]
        td, r = tds[exp], float(rr[i])
        T = td / 252
        s_adj = float(mt["spot"])
        F = _parity_forward(ch, T, r, s_adj)
        src = "parity"
        if F is None or abs(F / (s_adj * math.exp(r * T)) - 1) > a.max_parity_gap:
            F, src = s_adj * math.exp(r * T), "adjusted close"
        else:
            n_parity += 1
        spot = F * math.exp(-r * T)
        rows = []
        for o in ch.itertuples():
            if (o.kind == "put" and o.strike >= F) or (o.kind == "call" and o.strike < F):
                continue  # OTM side only, relative to the forward
            rows.append(dict(kind=o.kind, strike=float(o.strike), bid=None, ask=None, mid=float(o.close), last=float(o.close),
                             high=float(o.high), low=float(o.low), volume=float(o.volume),
                             transactions=float(getattr(o, "transactions", float("nan"))), source="day_close"))
        if not rows:
            skipped["no OTM trades"] = skipped.get("no OTM trades", 0) + 1
            continue
        quotes.append(dict(ticker=tk, date=d, spot=spot, spot_adjusted_close=s_adj, spot_source=src, expiry=str(exp),
                           trading_days=td, r=r, eval_index=i, options=sorted(rows, key=lambda o: o["strike"])))
        keep.append(i)
    if not keep:
        raise SystemExit(f"no usable contexts (skipped: {skipped}). Did `download` run for this export?")
    os.makedirs(a.out, exist_ok=True)
    keep = np.array(keep)
    np.save(os.path.join(a.out, "eval_ctx.npy"), ctx[keep].astype(np.float32))
    np.save(os.path.join(a.out, "eval_realized.npy"), real[keep].astype(np.float32))
    json.dump(quotes, open(os.path.join(a.out, "quotes.json"), "w"))
    out_cfg = dict(cfg, kind="market", source="massive day_aggs_v1 (day close, no bid/ask)", n_eval=int(len(keep)),
                   source_export=os.path.abspath(a.export), tickers=sorted(set(q["ticker"] for q in quotes)),
                   dates=[quotes[0]["date"], quotes[-1]["date"]])
    json.dump(out_cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    for fn in ("train.npy", "val.npy"):  # so baselines that fit on training data (GARCH) work from this folder
        src, dst = os.path.join(a.export, fn), os.path.join(a.out, fn)
        if os.path.exists(src) and not os.path.lexists(dst):
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                shutil.copy(src, dst)
    n_opt = [len(q["options"]) for q in quotes]
    per_tk = {t: sum(q["ticker"] == t for q in quotes) for t in out_cfg["tickers"]}
    print(f"built {a.out}: {len(keep)} contexts ({per_tk}), {out_cfg['dates'][0]} .. {out_cfg['dates'][1]}, "
          f"median {int(np.median(n_opt))} traded OTM options each, spot from parity in {n_parity}/{len(keep)}"
          + (f"; skipped {skipped}" if skipped else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--export", required=True); d.add_argument("--out", default="bench_data/massive_raw")
    d.add_argument("--since", default=None, help="first date to fetch (default: 2 years back, the Starter window)")
    d.add_argument("--source", default="auto", choices=["auto", "s3", "rest"])
    d.add_argument("--min_days", type=int, default=5, help="REST: shortest expiry, as in build")
    d.add_argument("--strike_band", type=float, default=0.35, help="REST: strikes within +-35%% of spot")
    d.add_argument("--workers", type=int, default=8, help="REST: parallel requests for daily bars")
    b = sub.add_parser("build")
    b.add_argument("--export", required=True); b.add_argument("--raw", default="bench_data/massive_raw")
    b.add_argument("--out", required=True)
    b.add_argument("--min_days", type=int, default=5, help="shortest expiry to use, in trading days")
    b.add_argument("--min_contracts", type=int, default=6, help="traded contracts needed on an expiry to use it")
    b.add_argument("--max_parity_gap", type=float, default=0.03, help="fall back to the adjusted close if parity is off by more")
    a = ap.parse_args(argv)
    download(a) if a.cmd == "download" else build(a)


if __name__ == "__main__":
    main()
