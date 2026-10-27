"""B4: martingale Schroedinger bridge between two maturities (after Henry-Labordere 2019), on a 1-D grid.

Vanillas at 20d and 60d pin down the marginals mu (of S_20) and nu (of S_60) but not how they are joined.
We look for the coupling pi closest in KL to a reference transition kernel R(S_20 -> S_60) such that
    pi has marginals mu and nu,   and   E_pi[S_60 | S_20] = S_20   (martingale).
The optimum has the form pi_ij = exp(a_i + b_j + D_i (y_j - x_i)) R_ij, solved here by a Sinkhorn-type loop:
update a (row marginals), update b (column marginals), Newton-update each D_i (row martingale condition).

Then price forward-start options on R = S_60/S_20 under pi and compare with the true D2 prices:
the ATM straddle E|R - 1| (mostly a forward-ATM-vol test, easy) and the 90% put / 110% call
(forward-skew tests, where dynamics actually matter).
Hypothesis H11: every coupling here matches the vanillas, but the bridge with a sensible reference lands
much closer to the true forward-start price than naive alternatives. Comparators:
  * fwd_black  : Black forward-start straddle at the forward vol implied by the two ATM vols (no smile)
  * independent: product coupling mu x nu (matches marginals, not a martingale)
  * sb_flat    : bridge with a flat 20% lognormal reference
  * sb_fwdvol  : bridge with a lognormal reference at the ATM forward vol (uses vanilla info only)
Marginals come from the true D2 densities by default (clean test), or from a B1 model's predicted densities
(--density_pred results/trackB/preds/ebm_per_mat5.npz) for the end-to-end version.
TODO for the paper: Dupire local vol as the classic comparator (needs a smooth full surface).

  python -m starters.b4_schrodinger_bridge --test bench_data/D2/test.npz
"""
import os as _os
import sys as _sys

if __package__ in (None, ""):  # also allow `python path/to/file.py`, not only `python -m pkg.file`
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import math
import os

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from bench.surfaces import TAUS_D
from starters.common import load_D2

SIG_REF = 0.2


def grid_marginal(dens_z, z, tau, zmax=8.0):
    keep = np.abs(z) <= zmax
    x = z[keep] * SIG_REF * math.sqrt(tau)
    w = np.maximum(dens_z[keep], 0)
    w = w / w.sum()
    S = np.exp(x)
    S = S / (w * S).sum()  # forward exactly 1
    return S, w


def lognormal_kernel_log(S1, S2, sigma, dt):
    m = -0.5 * sigma ** 2 * dt
    sd = sigma * math.sqrt(dt)
    lr = np.log(S2)[None, :] - np.log(S1)[:, None]
    return -0.5 * ((lr - m) / sd) ** 2


def martingale_sb(S1, mu, S2, nu, logR, iters=2000, newton=3, tol=1e-9):
    dy = S2[None, :] - S1[:, None]
    a = np.zeros(len(S1)); b = np.zeros(len(S2)); D = np.zeros(len(S1))
    lmu, lnu = np.log(mu + 1e-300), np.log(nu + 1e-300)
    for it in range(iters):
        L = a[:, None] + b[None, :] + D[:, None] * dy + logR
        a += lmu - logsumexp(L, axis=1)
        L = a[:, None] + b[None, :] + D[:, None] * dy + logR
        b += lnu - logsumexp(L, axis=0)
        for _ in range(newton):
            L = b[None, :] + D[:, None] * dy + logR
            W = np.exp(L - L.max(1, keepdims=True)); W /= W.sum(1, keepdims=True)
            m1 = (W * dy).sum(1); m2 = (W * dy ** 2).sum(1) - m1 ** 2
            D -= np.clip(m1 / np.maximum(m2, 1e-12), -50, 50)
        if it % 50 == 0:
            L = a[:, None] + b[None, :] + D[:, None] * dy + logR
            pi = np.exp(L)
            err = max(np.abs(pi.sum(1) - mu).max(), np.abs(pi.sum(0) - nu).max(),
                      np.abs((pi * dy).sum(1)).max())
            if err < tol:
                break
    pi = np.exp(a[:, None] + b[None, :] + D[:, None] * dy + logR)
    diag = dict(iters=it + 1, row_err=float(np.abs(pi.sum(1) - mu).max()), col_err=float(np.abs(pi.sum(0) - nu).max()),
                mart_err=float(np.abs((pi * dy).sum(1)).sum()))  # mass-weighted |E[S2 - S1 | S1]|
    return pi, diag


PAYOFFS = {"fwd_straddle_20_60": lambda R: np.abs(R - 1),
           "fwd_put_20_60_k90": lambda R: np.maximum(0.9 - R, 0),
           "fwd_call_20_60_k110": lambda R: np.maximum(R - 1.1, 0)}


def black_fwd(v):
    """Black prices on R = S60/S20 (lognormal, forward 1, total vol v) for each payoff in PAYOFFS."""
    def call(K):
        d1 = (-math.log(K) + 0.5 * v * v) / v
        return norm.cdf(d1) - K * norm.cdf(d1 - v)
    c1 = call(1.0)
    return {"fwd_straddle_20_60": 2 * c1, "fwd_put_20_60_k90": call(0.9) - (1 - 0.9), "fwd_call_20_60_k110": call(1.1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="bench_data/D2/test.npz")
    ap.add_argument("--density_pred", default=None, help="npz with a 'density' array (e.g. from b1_ebm); default = true densities")
    ap.add_argument("--n_surfaces", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2, help="grid subsampling for speed")
    ap.add_argument("--out", default="results/trackB")
    args = ap.parse_args(argv)
    te = load_D2(args.test)
    j1, j2 = list(TAUS_D).index(20), list(TAUS_D).index(60)
    t1, t2 = TAUS_D[j1] / 252, TAUS_D[j2] / 252
    dt = t2 - t1
    dens = te["density"] if args.density_pred is None else np.load(args.density_pred)["density"]
    z = te["zgrid"][::args.stride]
    gi = np.nonzero(te["family"] == "garch")[0][:args.n_surfaces]
    a_idx = np.argmin(np.abs(te["ks"]))
    rows = []
    for s in gi:
        S1, mu = grid_marginal(dens[s, j1, ::args.stride], z, t1)
        S2, nu = grid_marginal(dens[s, j2, ::args.stride], z, t2)
        s1, s2 = te["iv"][s, j1, a_idx], te["iv"][s, j2, a_idx]
        sfwd = math.sqrt(max(s2 ** 2 * t2 - s1 ** 2 * t1, 1e-8) / dt)
        names = list(te["exotic_names"])
        truth = {p: float(te["exotics"][s, names.index(p)]) for p in PAYOFFS}
        out = dict(surface=int(s), truth=truth, fwd_vol=sfwd, prices={})
        v = sfwd * math.sqrt(dt)
        out["prices"]["fwd_black"] = black_fwd(v)
        ratio = S2[None, :] / S1[:, None]
        out["prices"]["independent"] = {p: float((mu[:, None] * nu[None, :] * f(ratio)).sum()) for p, f in PAYOFFS.items()}
        for name, sig in (("sb_flat", SIG_REF), ("sb_fwdvol", sfwd)):
            pi, diag = martingale_sb(S1, mu, S2, nu, lognormal_kernel_log(S1, S2, sig, dt))
            out["prices"][name] = {p: float((pi * f(ratio)).sum()) for p, f in PAYOFFS.items()}
            out[name + "_diag"] = diag
        rows.append(out)
        print(f"  surface {s} (mart err {out['sb_fwdvol_diag']['mart_err']:.1e}): " + "  ".join(
            f"{p}: truth {truth[p]:.4f} black {out['prices']['fwd_black'][p]:.4f} sb {out['prices']['sb_fwdvol'][p]:.4f}"
            for p in PAYOFFS), flush=True)
    summ = {}
    for k in ("fwd_black", "independent", "sb_flat", "sb_fwdvol"):
        summ[k] = {p: float(np.mean([abs(r["prices"][k][p] - r["truth"][p]) / r["truth"][p] for r in rows])) for p in PAYOFFS}
    print("mean |rel err| by payoff:")
    for k, v in summ.items():
        print(f"  {k:12s} " + "  ".join(f"{p}={e:.1%}" for p, e in v.items()))
    os.makedirs(args.out, exist_ok=True)
    tag = "true_marginals" if args.density_pred is None else os.path.splitext(os.path.basename(args.density_pred))[0]
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(args.out, f"b4_sb_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
