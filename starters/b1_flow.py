"""B1: normalizing-flow risk-neutral density, fitted per surface and maturity to its quotes.

1-D monotone flow from a standard normal:  x = s_hat sqrt(tau) * f(u) + mu,  u ~ N(0, 1),
    f(u) = a u + sum_k w_k tanh(b_k u + c_k),   a, w_k, b_k > 0   (so f is strictly increasing)
mu is set analytically so E[e^x] = 1 (exact forward). s_hat = median quoted IV at that maturity (or nearest one).
Prices come from a fixed quantile grid of u (deterministic, low noise). Loss per maturity:
    mean squared (price error / quote noise)  +  reg * ||flow - identity||^2
so with no or few quotes it falls back to a lognormal at s_hat. All (surface, maturity) problems are fitted
at once with vmap. Exact likelihood and exact density (change of variables) for the B1 density metric.

  python -m starters.b1_flow --test bench_data/D2/test.npz
"""
import os as _os
import sys as _sys

if __package__ in (None, ""):  # also allow `python path/to/file.py`, not only `python -m pkg.file`
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import norm

from bench.finance import bs_otm, bs_vega
from bench.trackB import save_pred, score_and_save
from starters.common import Timer, density_to_iv, load_D2

SIG_REF = 0.2


def f_and_grad(p, u):
    a = jax.nn.softplus(p["a"]) + 1e-3
    w = jax.nn.softplus(p["w"]); b = jax.nn.softplus(p["b"])
    t = jnp.tanh(b[:, None] * u[None, :] + p["c"][:, None])
    f = a * u + (w[:, None] * t).sum(0)
    df = a + (w[:, None] * b[:, None] * (1 - t ** 2)).sum(0)
    return f, df


def init_params(K, key):
    return dict(a=jnp.array(np.log(np.e - 1), jnp.float32),  # softplus^-1(1): starts as identity
                w=jnp.full((K,), -6.0), b=jnp.zeros((K,)), c=0.5 * jax.random.normal(key, (K,)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="bench_data/D2/test.npz")
    ap.add_argument("--K", type=int, default=8, help="tanh units in the flow")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--reg", type=float, default=0.05)
    ap.add_argument("--n_quad", type=int, default=4000)
    ap.add_argument("--scenarios", nargs="+", default=["per_mat5", "sparse10", "nowings"])
    ap.add_argument("--pred_dir", default="results/trackB/preds")
    ap.add_argument("--out", default="results/trackB")
    args = ap.parse_args(argv)
    te = load_D2(args.test)
    S, m, n = te["iv"].shape
    taus, xg, ks = te["taus"], te["xgrid"], te["ks"]
    u = jnp.asarray(norm.ppf((np.arange(args.n_quad) + 0.5) / args.n_quad), jnp.float32)
    lt = np.log(taus)

    for sc in args.scenarios:
        Mk, Qt = te[f"mask_{sc}"], te[f"quote_{sc}"]
        shat = np.empty((S, m))
        for s in range(S):
            for j in range(m):
                if Mk[s, j].any():
                    shat[s, j] = np.median(Qt[s, j][Mk[s, j]])
                else:
                    js = np.nonzero(Mk[s].any(1))[0]
                    jn = js[np.argmin(np.abs(lt[js] - lt[j]))]
                    shat[s, j] = np.median(Qt[s, jn][Mk[s, jn]])
        q = np.nan_to_num(Qt, nan=0.2)
        price = bs_otm(q, xg[None], taus[None, :, None])
        nz = np.maximum(bs_vega(q, xg[None], taus[None, :, None]) * 0.005 * (1 + np.abs(ks) / 2), 1e-7)
        # flatten (surface, maturity) into problems
        P = S * m
        scale = jnp.asarray((shat * np.sqrt(taus)[None]).reshape(P), jnp.float32)
        K_ = jnp.asarray(np.exp(np.broadcast_to(xg[None], (S, m, n))).reshape(P, n), jnp.float32)
        isput = jnp.asarray(np.broadcast_to(xg[None] < 0, (S, m, n)).reshape(P, n))
        mask = jnp.asarray(Mk.reshape(P, n).astype(np.float32))
        pr = jnp.asarray(price.reshape(P, n), jnp.float32)
        nzj = jnp.asarray(nz.reshape(P, n), jnp.float32)
        params = jax.vmap(lambda k: init_params(args.K, k))(jax.random.split(jax.random.PRNGKey(0), P))

        def x_of(p, sc_):
            f, df = f_and_grad(p, u)
            x = sc_ * f
            mu = -jnp.log(jnp.mean(jnp.exp(x)))
            return x + mu, df

        def loss_one(p, sc_, K, isp, mk, pq, nzq):
            x, _ = x_of(p, sc_)
            S_ = jnp.exp(x)
            pm = jnp.where(isp[:, None], jnp.maximum(K[:, None] - S_[None], 0), jnp.maximum(S_[None] - K[:, None], 0)).mean(-1)
            fit = jnp.sum(mk * ((pm - pq) / nzq) ** 2) / jnp.maximum(mk.sum(), 1.0)
            reg = (jax.nn.softplus(p["a"]) - 1) ** 2 + jnp.sum(jax.nn.softplus(p["w"]) ** 2)
            return fit + args.reg * reg

        opt = optax.adam(args.lr)
        st = jax.vmap(opt.init)(params)

        @jax.jit
        def step(params, st):
            l, g = jax.vmap(jax.value_and_grad(loss_one))(params, scale, K_, isput, mask, pr, nzj)
            upd, st = jax.vmap(opt.update)(g, st, params)
            return jax.vmap(optax.apply_updates)(params, upd), st, l

        clock = Timer()
        for it in range(args.steps):
            params, st, l = step(params, st)
        print(f"  flow/{sc}: fitted {P} maturity slices in {clock():.0f}s, mean loss {float(l.mean()):.3f}", flush=True)

        # exact density on the z grid by inverting the monotone map (bisection), change of variables
        zg = te["zgrid"]
        xgrid_d = jnp.asarray((zg[None, :] * SIG_REF * np.sqrt(taus)[:, None]), jnp.float32)  # [m, G]

        def dens_one(p, sc_, xd):
            f, _ = f_and_grad(p, u)
            mu = -jnp.log(jnp.mean(jnp.exp(sc_ * f)))
            target = (xd - mu) / sc_
            lo, hi = jnp.full_like(target, -12.0), jnp.full_like(target, 12.0)
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                fm, _ = f_and_grad(p, mid)
                lo, hi = jnp.where(fm < target, mid, lo), jnp.where(fm < target, hi, mid)
            uu = 0.5 * (lo + hi)
            _, dfu = f_and_grad(p, uu)
            px = jnp.exp(-0.5 * uu ** 2) / jnp.sqrt(2 * jnp.pi) / (dfu * sc_)
            return px

        xd_rep = jnp.asarray(np.repeat(np.asarray(xgrid_d)[None], S, 0).reshape(P, -1))
        px = np.asarray(jax.jit(jax.vmap(dens_one))(params, scale, xd_rep)).reshape(S, m, -1)
        dens = px * (SIG_REF * np.sqrt(taus))[None, :, None]  # to z units
        dens = dens / (dens.sum(-1, keepdims=True) * (zg[1] - zg[0]))
        iv = density_to_iv(dens, zg, xg, taus)
        path = os.path.join(args.pred_dir, f"flow_{sc}.npz")
        save_pred(path, "flow", sc, iv, density=dens)
        score_and_save(args.test, path, args.out)


if __name__ == "__main__":
    main()
