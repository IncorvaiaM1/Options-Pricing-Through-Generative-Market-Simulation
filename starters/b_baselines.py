"""Track B baselines. Every Track B claim is measured against these.

  linear2d : linear interpolation of IV in (k, log tau), nearest-neighbour fill outside the hull
  ssvi     : surface SVI (Gatheral & Jacquier 2014), theta(tau) = (s0 + s1 ln(tau/20d))^2 tau, phi = eta/sqrt(theta)
  gp       : Gaussian process on IV over (k, log tau), anisotropic RBF + white noise; gives a 90% band
  heston   : Heston calibrated to the quotes (characteristic-function pricer)
  maxent   : Buchen & Kelly (1996) maximum entropy density per maturity, lognormal prior, matching the forward
             exactly and each quoted price up to its noise level (relaxed max-ent: an L2 penalty on the dual
             variables, scaled by vega x quote-noise); exact matching of noisy quotes overfits badly.
             This is the classical B1 density baseline (also gives a surface).
  ssvi_bl  : Breeden-Litzenberger density from the SSVI fit (second baseline for B1 densities)

  python -m starters.b_baselines --test bench_data/D2/test.npz --scenarios per_mat5 sparse10 nowings
  python -m bench.trackB table "results/trackB/*.json"
"""
import os as _os
import sys as _sys

if __package__ in (None, ""):  # also allow `python path/to/file.py`, not only `python -m pkg.file`
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import math
import os
import time
import warnings

import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import least_squares, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from bench.finance import bs_call, bs_otm, bs_vega, heston_otm, implied_vol
from bench.trackB import save_pred, score_and_save
from starters.common import density_to_iv, load_D2

warnings.filterwarnings("ignore")
SIG_REF = 0.2


def quotes_of(test, sc, s):
    m = test[f"mask_{sc}"][s]
    j, i = np.nonzero(m)
    return j, i, test[f"quote_{sc}"][s][m]


# ---------------------------------------------------------------- linear
def fit_linear(test, sc, s):
    j, i, q = quotes_of(test, sc, s)
    ks, lt = test["ks"], np.log(test["taus"])
    pts = np.c_[ks[i], lt[j]]
    K, T = np.meshgrid(ks, lt)
    lin = griddata(pts, q, (K, T), method="linear") if len(q) >= 3 else np.full(K.shape, np.nan)
    near = griddata(pts, q, (K, T), method="nearest")
    return np.where(np.isfinite(lin), lin, near)


# ---------------------------------------------------------------- SSVI
def ssvi_iv(p, x, tau):
    s0, s1, eta, rho = p
    atm = np.maximum(s0 + s1 * np.log(tau / (20 / 252)), 1e-3)
    theta = atm ** 2 * tau
    phi = eta / np.sqrt(theta)
    w = theta / 2 * (1 + rho * phi * x + np.sqrt((phi * x + rho) ** 2 + 1 - rho ** 2))
    return np.sqrt(np.maximum(w, 1e-10) / tau)


def fit_ssvi(test, sc, s):
    j, i, q = quotes_of(test, sc, s)
    x, tau = test["xgrid"][j, i], test["taus"][j]
    best = None
    for rho0 in (-0.5, 0.0):
        p0 = [np.median(q), 0.0, 0.5, rho0]
        r = least_squares(lambda p: (ssvi_iv(p, x, tau) - q) * 100, p0,
                          bounds=([0.02, -0.5, 0.01, -0.99], [2.0, 0.5, 4.0, 0.99]))
        if best is None or r.cost < best.cost:
            best = r
    return ssvi_iv(best.x, test["xgrid"], test["taus"][:, None]), best.x


def bl_density(iv_fn, test):
    """Breeden-Litzenberger: density of x from d2C/dK2 on the z grid of each maturity, in z units."""
    z = test["zgrid"]
    out = np.empty((len(test["taus"]), len(z)))
    for j, t in enumerate(test["taus"]):
        sd = SIG_REF * math.sqrt(t)
        x = z * sd
        K = np.exp(x)
        C = bs_call(iv_fn(x, t), x, t)
        d2 = np.gradient(np.gradient(C, K), K)
        p = np.maximum(d2 * K * sd, 0)  # density in z units
        out[j] = p / (p.sum() * (z[1] - z[0]) + 1e-300)
    return out


# ---------------------------------------------------------------- GP
def fit_gp(test, sc, s):
    j, i, q = quotes_of(test, sc, s)
    Xq = np.c_[test["ks"][i], np.log(test["taus"][j])]
    mu = q.mean()
    k = ConstantKernel(1e-3, (1e-6, 1)) * RBF([2.0, 1.0], [(0.3, 30), (0.1, 10)]) + WhiteKernel(1e-5, (1e-8, 1e-2))
    gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=2, random_state=0).fit(Xq, q - mu)
    K, T = np.meshgrid(test["ks"], np.log(test["taus"]))
    m, sd = gp.predict(np.c_[K.ravel(), T.ravel()], return_std=True)
    m = m.reshape(K.shape) + mu
    sd = sd.reshape(K.shape)
    return m, m - 1.645 * sd, m + 1.645 * sd


# ---------------------------------------------------------------- Heston
def fit_heston(test, sc, s):
    j, i, q = quotes_of(test, sc, s)
    xg, taus = test["xgrid"], test["taus"]
    mats = np.unique(j)

    def model_iv(p):
        out = np.empty_like(q)
        for jj in mats:
            sel = j == jj
            pr = heston_otm(xg[jj, i[sel]], taus[jj], *p)
            out[sel] = implied_vol(pr, xg[jj, i[sel]], taus[jj])
        return out

    def resid(p):
        r = (model_iv(p) - q) * 100
        return np.where(np.isfinite(r), r, 50.0)

    v = np.median(q) ** 2
    best = None
    for p0 in ([v, 2.0, v, 0.5, -0.5], [v, 4.0, v, 1.0, -0.8]):
        r = least_squares(resid, p0, bounds=([1e-3, 0.1, 1e-3, 0.05, -0.99], [1.0, 10.0, 1.0, 2.0, 0.5]), max_nfev=150)
        if best is None or r.cost < best.cost:
            best = r
    iv = np.empty(xg.shape)
    for jj, t in enumerate(taus):
        iv[jj] = implied_vol(heston_otm(xg[jj], t, *best.x), xg[jj], t)
    return iv, best.x


# ---------------------------------------------------------------- max entropy (Buchen & Kelly)
def fit_maxent(test, sc, s, noise_vp=0.5):
    j, i, q = quotes_of(test, sc, s)
    z = test["zgrid"]
    dz = z[1] - z[0]
    taus, xg = test["taus"], test["xgrid"]
    lt = np.log(taus)
    dens = np.empty((len(taus), len(z)))
    atm_all = np.median(q)
    for jj, t in enumerate(taus):
        sd_ref = SIG_REF * math.sqrt(t)
        x = z * sd_ref
        sel = j == jj
        # prior: lognormal at an ATM vol taken from the nearest quoted maturity
        if sel.any():
            a = np.median(q[sel])
        else:
            near = j[np.argmin(np.abs(lt[j] - lt[jj]))]
            a = np.median(q[j == near]) if len(q) else atm_all
        prior = np.exp(-0.5 * ((x + 0.5 * a ** 2 * t) / (a * math.sqrt(t))) ** 2)
        prior /= prior.sum()
        H = [np.exp(x) - 1]  # forward constraint E[S] = 1 (exact)
        c = [0.0]
        noise = [0.0]
        for ii, qq in zip(i[sel], q[sel]):
            kx = xg[jj, ii]
            H.append(np.maximum(np.exp(kx) - np.exp(x), 0) if kx < 0 else np.maximum(np.exp(x) - np.exp(kx), 0))
            c.append(float(bs_otm(qq, kx, t)))
            noise.append(float(bs_vega(qq, kx, t)) * noise_vp / 100 * (1 + abs(test["ks"][ii]) / 2))
        H = np.array(H); c = np.array(c)
        scale = np.maximum(np.abs(H).max(1), 1e-12)
        Hs, cs, ns = H / scale[:, None], c / scale, np.array(noise) / scale

        def dual(lam):  # relaxed max-ent dual: allows each price an error of about its noise level
            e = lam @ Hs
            m = e.max()
            w = prior * np.exp(e - m)
            Z = w.sum()
            val = m + np.log(Z) - lam @ cs + 0.5 * np.sum((ns * lam) ** 2)
            grad = (Hs * w).sum(1) / Z - cs + ns ** 2 * lam
            return val, grad

        r = minimize(dual, np.zeros(len(cs)), jac=True, method="L-BFGS-B")
        e = r.x @ Hs
        w = prior * np.exp(e - e.max())
        dens[jj] = w / (w.sum() * dz)
    return dens


METHODS = ("linear2d", "ssvi", "gp", "heston", "maxent")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="bench_data/D2/test.npz")
    ap.add_argument("--scenarios", nargs="+", default=["per_mat5", "sparse10", "nowings"])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--pred_dir", default="results/trackB/preds")
    ap.add_argument("--out", default="results/trackB")
    args = ap.parse_args(argv)
    test = load_D2(args.test)
    S = test["iv"].shape[0]
    for sc in args.scenarios:
        for meth in args.methods:
            t0 = time.time()
            ivs, lo, hi, dens = [], [], [], []
            for s in range(S):
                if meth == "linear2d":
                    ivs.append(fit_linear(test, sc, s))
                elif meth == "ssvi":
                    iv, p = fit_ssvi(test, sc, s)
                    ivs.append(iv)
                    dens.append(bl_density(lambda x, t: ssvi_iv(p, x, t), test))
                elif meth == "gp":
                    m, l, h = fit_gp(test, sc, s)
                    ivs.append(m); lo.append(l); hi.append(h)
                elif meth == "heston":
                    ivs.append(fit_heston(test, sc, s)[0])
                elif meth == "maxent":
                    d = fit_maxent(test, sc, s)
                    dens.append(d)
                    ivs.append(density_to_iv(d, test["zgrid"], test["xgrid"], test["taus"]))
            extra = {}
            if lo:
                extra.update(iv_lo=np.array(lo), iv_hi=np.array(hi))
            if dens:
                extra["density"] = np.array(dens)
            name = "ssvi" if meth == "ssvi" else meth
            path = os.path.join(args.pred_dir, f"{name}_{sc}.npz")
            save_pred(path, name, sc, np.array(ivs), **extra)
            print(f"  {meth}/{sc}: {time.time() - t0:.1f}s", flush=True)
            score_and_save(args.test, path, args.out)


if __name__ == "__main__":
    main()
