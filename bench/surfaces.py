"""D2: synthetic option surfaces with exact truth, for every Track B starter.

Two families, each surface with its own randomly drawn parameters:
  garch  : GJR-GARCH(1,1) (leverage term gives equity-like skew) with Student-t shocks (truncated at |eps| <= 15 so the MGF exists and calls are finite),
           started from a random state; priced by Monte Carlo with the empirical martingale correction
           (Duan & Simonato 1998) so the forward is exactly 1 at every date. Gives true prices, true
           risk-neutral densities (histograms) and true exotic prices.
  heston : Heston with random parameters; vanillas from the characteristic function, exotics by MC.

Grid: maturities TAUS_D days; strikes in std units k (KS), log strike x = k * 0.2 * sqrt(tau).
Quote scenarios (identical for every model, stored in the test file):
  per_mat5 : 5 random strikes per maturity (25 quotes)
  sparse10 : 10 random points over the whole surface
  nowings  : every strike with |k| <= 1.5 at every maturity; wings hidden
  full     : every grid point (used for B2 calibration)
Quotes carry IV noise with sd = 0.5 vol pts * (1 + |k|/2).

  python -m bench.surfaces --out bench_data/D2 --n_train 512 --n_test 64 --paths 20000
"""
import argparse
import os
import time

import numpy as np

from .finance import (EXOTICS, exotic_prices, heston_paths, heston_surface_iv, implied_vol,
                      otm_from_logret)

TAUS_D = np.array([5, 10, 20, 40, 60])
TAUS = TAUS_D / 252
KS = np.linspace(-3, 3, 13)
SIG_REF = 0.20
XGRID = KS[None, :] * SIG_REF * np.sqrt(TAUS)[:, None]          # [m, n] log strikes
ZGRID = np.linspace(-10, 10, 401)                                # density grid in std units (per maturity)
SCENARIOS = ("per_mat5", "sparse10", "nowings", "full")


def density_xgrid():
    """[m, G] log-return grid for densities: x = z * SIG_REF * sqrt(tau)."""
    return ZGRID[None, :] * SIG_REF * np.sqrt(TAUS)[:, None]


# ---------------------------------------------------------------- GARCH-t family
def _trunc_t(rng, nu, size, cap=15.0):
    s = np.sqrt(nu / (nu - 2))
    e = rng.standard_t(nu, size=size) / s
    bad = np.abs(e) > cap
    while bad.any():
        e[bad] = rng.standard_t(nu, size=bad.sum()) / s
        bad = np.abs(e) > cap
    return e


def sample_garch_params(rng):
    a = rng.uniform(0.02, 0.10)
    g = rng.uniform(0.0, 0.12)
    return dict(annual_vol=rng.uniform(0.12, 0.40), alpha=a, gamma=g,
                beta=rng.uniform(0.80, 0.975 - a - g / 2), nu=rng.uniform(3.5, 10.0))


def garch_paths(p, n_paths, rng, days=60, burn=300):
    var_lr = p["annual_vol"] ** 2 / 252
    a, g, b = p["alpha"], p["gamma"], p["beta"]
    omega = var_lr * (1 - a - g / 2 - b)
    v = var_lr
    for _ in range(burn):  # one burn-in path sets a realistic random starting state
        r = np.sqrt(v) * _trunc_t(rng, p["nu"], 1)[0]
        v = omega + (a + g * (r < 0)) * r ** 2 + b * v
    state = v
    var = np.full(n_paths, v)
    lnS = np.zeros(n_paths)
    out = np.zeros((n_paths, days + 1))
    for d in range(days):
        r = np.sqrt(var) * _trunc_t(rng, p["nu"], n_paths)
        lnS = lnS + r
        lnS = lnS - np.log(np.exp(lnS).mean())  # empirical martingale correction
        var = omega + (a + g * (r < 0)) * r ** 2 + b * var
        out[:, d + 1] = lnS
    return out, state


def sample_heston_params(rng):
    return (rng.uniform(0.01, 0.16), rng.uniform(0.5, 5.0), rng.uniform(0.01, 0.16),
            rng.uniform(0.1, 1.0), rng.uniform(-0.9, 0.0))


# ---------------------------------------------------------------- build
def build_surface(family, rng, n_paths, with_exotics):
    xd = density_xgrid()
    dz = ZGRID[1] - ZGRID[0]
    if family == "garch":
        p = sample_garch_params(rng)
        lnS, state = garch_paths(p, n_paths, rng)
        iv = np.empty(XGRID.shape)
        dens = np.empty((len(TAUS), len(ZGRID)))
        for j, (d, t) in enumerate(zip(TAUS_D, TAUS)):
            X = lnS[:, d]
            iv[j] = implied_vol(otm_from_logret(X, XGRID[j]), XGRID[j], t)
            z = X / (SIG_REF * np.sqrt(t))
            h, _ = np.histogram(z, bins=np.append(ZGRID - dz / 2, ZGRID[-1] + dz / 2))
            dens[j] = h / (h.sum() * dz)  # density in z units
        params = np.array([p["annual_vol"], p["alpha"], p["gamma"], p["beta"], p["nu"], np.sqrt(state * 252)])
        ex = exotic_prices(lnS) if with_exotics else None
        return iv, dens, params, ex
    hp = sample_heston_params(rng)
    iv = heston_surface_iv(XGRID, TAUS, hp)
    ex = exotic_prices(heston_paths(hp, 60, n_paths, rng)) if with_exotics else None
    return iv, np.full((len(TAUS), len(ZGRID)), np.nan), np.array(hp), ex


def make_quotes(iv, scenario, rng, noise_vp=0.5):
    m, n = iv.shape
    mask = np.zeros((m, n), bool)
    if scenario == "per_mat5":
        for j in range(m):
            mask[j, rng.choice(n, 5, replace=False)] = True
    elif scenario == "sparse10":
        mask.flat[rng.choice(m * n, 10, replace=False)] = True
    elif scenario == "nowings":
        mask[:, np.abs(KS) <= 1.5] = True
    elif scenario == "full":
        mask[:] = True
    mask &= np.isfinite(iv)
    sd = noise_vp / 100 * (1 + np.abs(KS)[None, :] / 2)
    q = np.where(mask, iv + sd * rng.standard_normal(iv.shape), np.nan)
    return mask, np.maximum(q, 0.01)


def build_split(n, rng, n_paths, with_exotics, garch_frac, label):
    fam = np.where(rng.random(n) < garch_frac, "garch", "heston")
    IV, DENS, EX = [], [], []
    params_g = np.full((n, 6), np.nan)  # annual_vol, alpha, gamma, beta, nu, state_vol
    params_h = np.full((n, 5), np.nan)
    t0 = time.time()
    for i in range(n):
        iv, dens, par, ex = build_surface(fam[i], rng, n_paths, with_exotics)
        IV.append(iv); DENS.append(dens)
        (params_g if fam[i] == "garch" else params_h)[i] = par
        if with_exotics:
            EX.append([ex[k] for k in EXOTICS])
        if (i + 1) % max(1, n // 10) == 0:
            print(f"  {label}: {i + 1}/{n} surfaces  {time.time() - t0:.0f}s", flush=True)
    out = dict(iv=np.array(IV), density=np.array(DENS), family=fam, garch_params=params_g,
               heston_params=params_h, taus=TAUS, taus_days=TAUS_D, ks=KS, xgrid=XGRID, zgrid=ZGRID)
    if with_exotics:
        out["exotics"] = np.array(EX)
        out["exotic_names"] = np.array(EXOTICS)
    return out


def load(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_data/D2")
    ap.add_argument("--n_train", type=int, default=512)
    ap.add_argument("--n_test", type=int, default=64)
    ap.add_argument("--paths", type=int, default=20000, help="MC paths per GARCH surface")
    ap.add_argument("--test_paths", type=int, default=100000, help="MC paths per test surface (truth quality)")
    ap.add_argument("--garch_frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train = build_split(args.n_train, rng, args.paths, False, args.garch_frac, "train")
    np.savez_compressed(os.path.join(args.out, "train.npz"), **train)

    test = build_split(args.n_test, rng, args.test_paths, True, args.garch_frac, "test")
    qrng = np.random.default_rng(args.seed + 1)
    for sc in SCENARIOS:
        masks, quotes = zip(*[make_quotes(iv, sc, qrng) for iv in test["iv"]])
        test[f"mask_{sc}"] = np.array(masks)
        test[f"quote_{sc}"] = np.array(quotes)
    np.savez_compressed(os.path.join(args.out, "test.npz"), **test)
    print(f"saved {args.out}/train.npz ({args.n_train}) and test.npz ({args.n_test}); "
          f"NaN IV cells train/test: {np.isnan(train['iv']).mean():.3%} / {np.isnan(test['iv']).mean():.3%}")


if __name__ == "__main__":
    main()
