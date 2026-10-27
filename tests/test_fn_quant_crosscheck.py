"""Cross-check: the benchmark's Black-Scholes (bench/finance.py) vs fatnarwhal-quant.

Two independent implementations (ours in normalized units, FatNarwhal's scalar tool code
in real units) must agree, so a pricing bug in either one shows up here.

  pip install fatnarwhal-quant   # or: pip install -e <path to fatnarwhal-quant>
  python -m pytest tests/test_fn_quant_crosscheck.py
"""
import numpy as np
import pytest

fnq = pytest.importorskip("fatnarwhal_quant")
from fatnarwhal_quant import normalized as fnn  # noqa: E402
from fatnarwhal_quant import options as fno  # noqa: E402

from bench import finance as bf  # noqa: E402

X = np.linspace(-0.4, 0.4, 17)
TAUS = [5 / 252, 20 / 252, 0.5, 1.0]
SIGMAS = [0.08, 0.2, 0.45]


@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("sigma", SIGMAS)
def test_otm_prices_match_vectorized(tau, sigma):
    np.testing.assert_allclose(bf.bs_otm(sigma, X, tau), fnn.otm_price(sigma, X, tau), rtol=1e-9, atol=1e-14)


@pytest.mark.parametrize("tau", TAUS)
def test_call_matches_scalar_site_tool(tau):
    # the site tool works in real units; S = 100, K = 100 e^x, r = q = 0 is the same option scaled by 100
    for x in X[::4]:
        site = fno.black_scholes(100.0, 100.0 * np.exp(x), tau, 0.2, 0.0, 0.0, "call")["price"] / 100.0
        assert float(bf.bs_call(0.2, x, tau)) == pytest.approx(site, rel=1e-9, abs=1e-14)


def test_vega_matches():
    for tau in TAUS:
        g = fnn.otm_greeks(0.25, X, tau)
        np.testing.assert_allclose(bf.bs_vega(0.25, X, tau), g["vega"], rtol=1e-9, atol=1e-14)


def test_implied_vol_agrees():
    tau = 20 / 252
    sig = 0.15 + 0.3 * np.abs(X)  # a smile
    px = bf.bs_otm(sig, X, tau)
    a = bf.implied_vol(px, X, tau)
    b = fnn.otm_implied_vol(px, X, tau)
    ok = np.isfinite(a)
    np.testing.assert_allclose(a[ok], b[ok], atol=1e-6)
    np.testing.assert_allclose(b[ok], sig[ok], atol=1e-6)
