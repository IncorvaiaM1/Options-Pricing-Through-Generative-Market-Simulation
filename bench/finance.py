"""Pricing utilities shared by Track B (and usable by Track A).

Conventions: spot = forward = 1, r = 0, no discounting. Strikes are log-moneyness x = ln K.
"OTM price" means a put for x < 0, a call for x >= 0 (the liquid side; IV is the same by parity).
Maturities are in years.
"""
import numpy as np
from scipy.special import ndtr


def bs_call(sigma, x, tau):
    K = np.exp(x)
    v = np.maximum(sigma, 1e-12) * np.sqrt(tau)
    d1 = (-x + 0.5 * v ** 2) / v
    return ndtr(d1) - K * ndtr(d1 - v)


def bs_otm(sigma, x, tau):
    c = bs_call(sigma, x, tau)
    p = c - (1 - np.exp(x))
    return np.where(x < 0, p, c)


def bs_vega(sigma, x, tau):
    v = np.maximum(sigma, 1e-12) * np.sqrt(tau)
    d1 = (-x + 0.5 * v ** 2) / v
    return np.sqrt(tau) * np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)


def implied_vol(price, x, tau, lo=1e-4, hi=4.0, iters=70):
    """Vectorized bisection on OTM prices. Returns nan where price is below the lower bound."""
    price, x, tau = np.broadcast_arrays(np.asarray(price, float), np.asarray(x, float), np.asarray(tau, float))
    a = np.full(price.shape, lo)
    b = np.full(price.shape, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        high = bs_otm(m, x, tau) > price
        b = np.where(high, m, b)
        a = np.where(high, a, m)
    iv = 0.5 * (a + b)
    floor = bs_otm(np.full(price.shape, lo), x, tau)
    return np.where(price <= floor * (1 + 1e-6) + 1e-14, np.nan, iv)


def otm_from_logret(X, x):
    """X: [..., N] samples of ln(S_T); x: [..., n] log strikes -> OTM prices [..., n]."""
    S = np.exp(X)[..., None, :]
    K = np.exp(x)[..., :, None]
    put = np.maximum(K - S, 0).mean(-1)
    call = np.maximum(S - K, 0).mean(-1)
    return np.where(x < 0, put, call)


# ---------------------------------------------------------------- Heston (characteristic function)
def heston_cf(u, tau, v0, kappa, theta, xi, rho):
    """CF of ln(S_T/S_0) under Heston with r=0 ("little trap" form). u may be complex."""
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b ** 2 + xi ** 2 * (iu + u ** 2))
    g = (b - d) / (b + d)
    e = np.exp(-d * tau)
    C = kappa * theta / xi ** 2 * ((b - d) * tau - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / xi ** 2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0)


_U = np.linspace(1e-6, 200.0, 2001)
_W = np.full_like(_U, _U[1] - _U[0]); _W[0] *= 0.5; _W[-1] *= 0.5


def heston_otm(x, tau, v0, kappa, theta, xi, rho):
    """OTM prices at log strikes x (1-D array) for one maturity, via Lewis (2001):
    C(K) = 1 - sqrt(K)/pi * int_0^inf Re[exp(-i u ln K) phi(u - i/2)] / (u^2 + 1/4) du."""
    x = np.atleast_1d(np.asarray(x, float))
    phi = heston_cf(_U - 0.5j, tau, v0, kappa, theta, xi, rho)
    integrand = np.real(np.exp(-1j * np.outer(x, _U)) * phi[None, :]) / (_U ** 2 + 0.25)
    call = 1 - np.exp(0.5 * x) / np.pi * (integrand * _W).sum(1)
    put = call - (1 - np.exp(x))
    return np.where(x < 0, put, call)


def heston_surface_iv(xgrid, taus, params):
    """xgrid [m, n] log strikes per maturity, taus [m] years -> IV [m, n]."""
    out = np.empty(xgrid.shape)
    for j, t in enumerate(taus):
        p = heston_otm(xgrid[j], t, *params)
        out[j] = implied_vol(p, xgrid[j], t)
    return out


def heston_paths(params, days, n_paths, rng, substeps=4, ems=True):
    """Euler full-truncation Heston, daily output of ln S [n_paths, days+1]; empirical martingale correction."""
    v0, kappa, theta, xi, rho = params
    dt = 1.0 / 252 / substeps
    lnS = np.zeros(n_paths)
    v = np.full(n_paths, v0)
    out = np.zeros((n_paths, days + 1))
    for d in range(days):
        for _ in range(substeps):
            z1 = rng.standard_normal(n_paths)
            z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_paths)
            vp = np.maximum(v, 0)
            lnS = lnS - 0.5 * vp * dt + np.sqrt(vp * dt) * z1
            v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
        if ems:
            lnS = lnS - np.log(np.exp(lnS).mean())
        out[:, d + 1] = lnS
    return out


# ---------------------------------------------------------------- exotics (on daily ln S paths)
EXOTICS = ("fwd_straddle_20_60", "asian_atm_call_60", "upout_call_60_b110", "downout_put_60_b90",
           "fwd_put_20_60_k90", "fwd_call_20_60_k110")


def exotic_prices(lnS):
    """lnS: [N, 61] daily log prices from day 0 to 60 (S0 = 1)."""
    S = np.exp(lnS)
    out = {}
    out["fwd_straddle_20_60"] = np.abs(S[:, 60] / S[:, 20] - 1).mean()
    out["asian_atm_call_60"] = np.maximum(S[:, 1:61].mean(1) - 1, 0).mean()
    alive_up = S[:, 1:61].max(1) < 1.10
    out["upout_call_60_b110"] = (np.maximum(S[:, 60] - 1, 0) * alive_up).mean()
    alive_dn = S[:, 1:61].min(1) > 0.90
    out["downout_put_60_b90"] = (np.maximum(1 - S[:, 60], 0) * alive_dn).mean()
    R = S[:, 60] / S[:, 20]  # forward-start options: sensitive to forward skew, i.e. to the dynamics
    out["fwd_put_20_60_k90"] = np.maximum(0.9 - R, 0).mean()
    out["fwd_call_20_60_k110"] = np.maximum(R - 1.1, 0).mean()
    return out
