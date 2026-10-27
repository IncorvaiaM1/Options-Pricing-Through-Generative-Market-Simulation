"""Price OTM options / straddles from model-generated paths and compare with the exact GARCH-t truth.

  python -m ardiff.evaluate --ckpts runs/diff/ckpt.pkl runs/tok/ckpt.pkl --out results/eval1

Strikes are set in units of the long-run H-day vol: K = S0 * exp(k * sd_H), k in --ks.
k < 0 prices a put, k > 0 a call, k = 0 a straddle. Errors are reported in price and in
implied-vol points (IV error is the fair way to compare deep OTM, where prices are tiny).
"""
import os as _os
import sys as _sys

if __package__ in (None, ""):  # also allow `python path/to/file.py`, not only `python -m pkg.file`
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import math
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from ardiff import data as D
from ardiff import diffusion as dif
from ardiff import tokens as tok
from ardiff.model import ARModel
from ardiff.train import build


# ------------------------------------------------------------------ sampling
def make_sampler(model, params, cfg, bins, decode_mode):
    L, H = cfg["ctx_len"], cfg["horizon"]
    sched = dif.make_schedule(cfg["diff_T"], cfg.get("diff_schedule", "linear"))

    @jax.jit
    def sample(ctx, key):  # ctx [N, L] normalized -> future [N, H]
        N = ctx.shape[0]
        buf = jnp.concatenate([ctx, jnp.zeros((N, H), ctx.dtype)], axis=1)

        def body(j, carry):
            buf, key = carry
            key, k = jax.random.split(key)
            z_all = model.apply({"params": params}, buf, False, method=ARModel.encode)
            z = jax.lax.dynamic_index_in_dim(z_all, L + j - 1, axis=1, keepdims=False)
            if cfg["head"] == "diffusion":
                eps_fn = lambda xt, t, zz: model.apply({"params": params}, xt, t, zz, method=ARModel.eps)
                x = dif.ddpm_sample(eps_fn, z, k, sched, cfg.get("diff_var", "large"))
            else:
                logits = model.apply({"params": params}, z, method=ARModel.logits)
                k1, k2 = jax.random.split(k)
                idx = jax.random.categorical(k1, logits)
                x = tok.decode(idx, bins, decode_mode, k2)
            buf = jax.lax.dynamic_update_index_in_dim(buf, x, L + j, axis=1)
            return buf, key

        buf, _ = jax.lax.fori_loop(0, H, body, (buf, key))
        return buf[:, L:]

    return sample


def generate(sampler, ctx_norm, n_paths, key, chunk):
    """Returns [E, n_paths, H] normalized future returns."""
    E, L = ctx_norm.shape
    rep = jnp.repeat(jnp.asarray(ctx_norm, jnp.float32), n_paths, axis=0)
    outs = []
    for s in range(0, rep.shape[0], chunk):
        key, k = jax.random.split(key)
        outs.append(np.asarray(sampler(rep[s:s + chunk], k)))
    return np.concatenate(outs, 0).reshape(E, n_paths, -1)


# ------------------------------------------------------------------ pricing
def otm_prices(logret, ks, sd_H, S0=100.0):
    """logret [E, M] terminal log returns -> price [E, len(ks)], se [E, len(ks)]."""
    ST = S0 * np.exp(logret)
    P, SE = [], []
    for k in ks:
        K = S0 * math.exp(k * sd_H)
        if k < 0:
            pay = np.maximum(K - ST, 0)
        elif k > 0:
            pay = np.maximum(ST - K, 0)
        else:
            pay = np.abs(ST - K)
        P.append(pay.mean(1))
        SE.append(pay.std(1) / math.sqrt(pay.shape[1]))
    return np.stack(P, 1), np.stack(SE, 1)


def _ncdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def bs_otm(sigma, ks, sd_H, T, S0=100.0):
    """Undiscounted Black-Scholes OTM put/call/straddle for array sigma [..., len(ks)]."""
    ks = np.asarray(ks, float)
    K = S0 * np.exp(ks * sd_H)
    v = np.maximum(sigma, 1e-8) * math.sqrt(T)
    d1 = (np.log(S0 / K) + 0.5 * v ** 2) / v
    d2 = d1 - v
    call = S0 * _ncdf(d1) - K * _ncdf(d2)
    put = call - (S0 - K)
    return np.where(ks < 0, put, np.where(ks > 0, call, call + put))


def implied_vol(price, ks, sd_H, T, lo=1e-4, hi=5.0, iters=80):
    price = np.asarray(price, float)
    a = np.full_like(price, lo)
    b = np.full_like(price, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        too_high = bs_otm(m, np.broadcast_to(ks, price.shape), sd_H, T) > price
        b = np.where(too_high, m, b)
        a = np.where(too_high, a, m)
    iv = 0.5 * (a + b)
    floor = bs_otm(np.full_like(price, lo), np.broadcast_to(ks, price.shape), sd_H, T)
    return np.where(price <= floor * 1.0001, np.nan, iv)


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--ks", type=float, nargs="+", default=[-4, -3, -2, -1, 0, 1, 2, 3, 4])
    ap.add_argument("--n_paths", type=int, default=4096, help="model paths per context")
    ap.add_argument("--n_mc_true", type=int, default=200000, help="truth paths per context")
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--token_decode", default="mean", choices=["mean", "center", "dequant"],
                    help="token head: decode to bin mean (default), bin center, or uniform jitter in bin")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    ckpts = [pickle.load(open(p, "rb")) for p in args.ckpts]
    labels = args.labels or [f"{c['cfg']['head']}" + (f"-{c['cfg']['n_bins']}{c['cfg']['bin_scheme'][0]}" if c['cfg']['head'] == 'token' else "") for c in ckpts]
    cfg0 = ckpts[0]["cfg"]
    for c in ckpts[1:]:
        for k in ("ctx_len", "horizon", "n_train", "n_val", "n_eval", "nu", "data_seed"):
            assert c["cfg"][k] == cfg0[k], f"checkpoints disagree on {k}"

    garch = D.GarchT(**ckpts[0]["garch"])
    splits = D.make_splits(garch, cfg0["ctx_len"], cfg0["horizon"], cfg0["n_train"], cfg0["n_val"], cfg0["n_eval"], cfg0["data_seed"])
    H = cfg0["horizon"]
    T = H / 252
    sd_H = garch.annual_vol * math.sqrt(T)
    ks = np.asarray(args.ks)

    # ---- truth
    rng = np.random.default_rng(args.seed)
    true_lr = D.true_terminal_logret(garch, splits["eval_next_var"], H, args.n_mc_true, rng)
    true_p, true_se = otm_prices(true_lr, ks, sd_H)
    true_iv = implied_vol(true_p, ks, sd_H, T)
    state_vol = np.sqrt(splits["eval_next_var"] * 252)
    stressed = state_vol >= np.quantile(state_vol, 2 / 3)

    # reference: Black-Scholes at long-run vol (ignores the GARCH state and fat tails)
    bs_p = bs_otm(np.full_like(true_p, garch.annual_vol), np.broadcast_to(ks, true_p.shape), sd_H, T)

    report = dict(ks=ks.tolist(), horizon=H, sd_H=sd_H, n_paths=args.n_paths, n_mc_true=args.n_mc_true,
                  state_annual_vol=state_vol.tolist(), truth=dict(price=true_p.tolist(), iv=true_iv.tolist(),
                  se=true_se.tolist()), models={})

    def summarize(name, p, iv):
        ive = np.abs(iv - true_iv) * 100  # vol points
        rel = (p - true_p) / true_p
        return dict(
            mean_abs_iv_err_by_k=np.nanmean(ive, 0).tolist(),
            mean_abs_iv_err_by_k_stressed=np.nanmean(ive[stressed], 0).tolist(),
            mean_rel_price_err_by_k=np.mean(rel, 0).tolist(),
            iv_nan_frac_by_k=np.mean(np.isnan(iv), 0).tolist(),
        )

    report["models"]["bs_longrun"] = dict(price=bs_p.tolist(), **summarize("bs", bs_p, implied_vol(bs_p, ks, sd_H, T)))

    key = jax.random.PRNGKey(args.seed)
    for lab, c in zip(labels, ckpts):
        cfg = c["cfg"]
        model = build(cfg)
        params = jax.tree_util.tree_map(jnp.asarray, c["params"])
        bins = None if c["bins"] is None else jax.tree_util.tree_map(jnp.asarray, c["bins"])
        if bins is not None and "means" not in bins:  # older checkpoints
            bins = tok.fit_bins(np.asarray(splits["train"] / c["scale"]), cfg["n_bins"], cfg["bin_scheme"], cfg["bin_clip"])
        sampler = make_sampler(model, params, cfg, bins, args.token_decode)
        ctx = splits["eval_ctx"] / c["scale"]
        key, k = jax.random.split(key)
        _ = generate(sampler, ctx[:1], 2, k, args.chunk)  # compile
        t0 = time.time()
        fut = generate(sampler, ctx, args.n_paths, k, args.chunk) * c["scale"]
        gen_s = time.time() - t0
        lr = fut.sum(-1)
        p, se = otm_prices(lr, ks, sd_H)
        iv = implied_vol(p, ks, sd_H, T)
        steps = fut.reshape(-1)
        s = summarize(lab, p, iv)
        s.update(price=p.tolist(), se=se.tolist(), iv=iv.tolist(), n_params=c["n_params"],
                 train_seconds=c["train_seconds"], gen_seconds=gen_s,
                 gen_us_per_path=1e6 * gen_s / (ctx.shape[0] * args.n_paths),
                 step_q999_abs=float(np.quantile(np.abs(steps), 0.999) / splits["scale"]),
                 step_max_abs=float(np.abs(steps).max() / splits["scale"]),
                 terminal_q=np.quantile(lr, [0.001, 0.01, 0.99, 0.999]).tolist())
        report["models"][lab] = s
        print(f"[{lab}] generated {ctx.shape[0]}x{args.n_paths} paths in {gen_s:.1f}s", flush=True)

    report["truth"]["terminal_q"] = np.quantile(true_lr, [0.001, 0.01, 0.99, 0.999]).tolist()

    # ---- table
    hdr = "| model | " + " | ".join(f"k={k:g}" for k in ks) + " |"
    sep = "|" + "---|" * (len(ks) + 1)
    lines = ["## Mean |IV error| (vol points), all contexts", "", hdr, sep]
    for name, m in report["models"].items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in m["mean_abs_iv_err_by_k"]) + " |")
    lines += ["", "## Mean |IV error| (vol points), stressed contexts (top third of state vol)", "", hdr, sep]
    for name, m in report["models"].items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in m["mean_abs_iv_err_by_k_stressed"]) + " |")
    lines += ["", "## Mean relative price error", "", hdr, sep]
    for name, m in report["models"].items():
        lines.append(f"| {name} | " + " | ".join(f"{v:+.1%}" for v in m["mean_rel_price_err_by_k"]) + " |")
    lines += ["", "## Cost and tails", "", "| model | params | train s | gen us/path | step 99.9% abs (std) | step max abs (std) | terminal q0.1% | terminal q99.9% |", "|---|---|---|---|---|---|---|---|"]
    for name, m in report["models"].items():
        if "n_params" in m:
            lines.append(f"| {name} | {m['n_params']:,} | {m['train_seconds']:.0f} | {m['gen_us_per_path']:.1f} | {m['step_q999_abs']:.1f} | {m['step_max_abs']:.1f} | {m['terminal_q'][0]:.4f} | {m['terminal_q'][3]:.4f} |")
    tq = report["truth"]["terminal_q"]
    tr = splits["train"].ravel()
    a = np.abs(tr) / splits["scale"]
    lines.append(f"| truth | | | | {np.quantile(a, 0.999):.1f} | {a.max():.1f} | {tq[0]:.4f} | {tq[3]:.4f} |")
    md = "\n".join(lines)
    print("\n" + md)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)

    try:
        _plot(report, labels, stressed, args.out)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print(f"saved {args.out}/report.md, report.json")


def _plot(report, labels, stressed, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = report["ks"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, m in report["models"].items():
        ax[0].plot(ks, m["mean_abs_iv_err_by_k"], marker="o", label=name)
    ax[0].set_xlabel("strike (std units of H-day vol)"); ax[0].set_ylabel("mean |IV error| (vol pts)")
    ax[0].set_title("Pricing error vs strike"); ax[0].legend(); ax[0].grid(alpha=.3)
    i = int(np.argmax(report["state_annual_vol"]))
    ax[1].plot(ks, np.array(report["truth"]["iv"][i]) * 100, "k-", lw=2, label="truth")
    for name in labels:
        ax[1].plot(ks, np.array(report["models"][name]["iv"][i]) * 100, marker="o", label=name)
    ax[1].set_xlabel("strike (std units)"); ax[1].set_ylabel("implied vol (%)")
    ax[1].set_title(f"Smile, most stressed context (state vol {report['state_annual_vol'][i]:.0%})")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "iv_error.png"), dpi=140)


if __name__ == "__main__":
    main()
