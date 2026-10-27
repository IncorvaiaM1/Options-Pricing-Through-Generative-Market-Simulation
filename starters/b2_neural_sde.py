"""B2: neural SDE calibrated to vanilla options, then used to price exotics (after Gierjatowicz et al.).

Risk-neutral by construction (r = 0):
    d ln S = -1/2 sigma^2 dt + sigma dW,     sigma = s0 * softplus(MLP(t, ln S / sqrt(t+), v))
    dv     = -kappa v dt + nu dB,            corr(dW, dB) = rho       (a learned stochastic-vol factor)
The log-price step is an exact martingale per step, so the forward is 1 by construction. Calibration fits
Monte Carlo prices (common random numbers) to the noisy full vanilla surface ("full" quotes), in
vega-scaled units (roughly IV error). An ensemble of independently initialized models gives a band for each
exotic price (H7: the band should cover the truth far more often than a single calibrated Heston).

Exotics (bench.finance.EXOTICS): forward-start straddle 20d->60d, Asian ATM call, up-and-out call (110%),
down-and-out put (90%). Truth comes from D2 (true GARCH / Heston dynamics). Baseline: Heston calibrated to the
same quotes, exotics by Heston Monte Carlo. Heston-family test surfaces are the control (Heston should win there).

  python -m starters.b2_neural_sde --test bench_data/D2/test.npz --n_surfaces 8
"""
import os as _os
import sys as _sys

if __package__ in (None, ""):  # also allow `python path/to/file.py`, not only `python -m pkg.file`
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bench.finance import EXOTICS, bs_otm, bs_vega, exotic_prices, heston_paths, implied_vol
from bench.trackB import save_pred, score_and_save
from starters.b_baselines import fit_heston
from starters.common import MLP, Timer, load_D2

DAYS = 60
DT = 1.0 / 252


class VolNet(nn.Module):
    width: int = 64

    @nn.compact
    def __call__(self, t, x, v):
        h = jnp.stack([t * 4.0, x / jnp.sqrt(t + 0.02), v], -1)
        return nn.softplus(MLP((self.width, self.width, 1))(h)[..., 0] + 0.5413)  # softplus(0.5413) = 1


class NSDE(nn.Module):
    width: int = 64

    def setup(self):
        self.vol = VolNet(self.width)
        self.raw = self.param("raw", lambda k: jnp.array([0.0, -1.0, -1.0]))  # kappa, nu, rho (transformed)

    def __call__(self, s0, dW, dZ):
        """dW, dZ: [DAYS, N] standard normals. Returns ln S [N, DAYS + 1]."""
        kappa = nn.softplus(self.raw[0]) * 5.0
        nu = nn.softplus(self.raw[1]) * 2.0
        rho = jnp.tanh(self.raw[2])
        N = dW.shape[1]
        x = jnp.zeros(N); v = jnp.zeros(N)
        xs = [x]
        for d in range(DAYS):
            t = jnp.full((N,), d * DT)
            sig = s0 * self.vol(t, x, v)
            x = x - 0.5 * sig ** 2 * DT + sig * jnp.sqrt(DT) * dW[d]
            dB = rho * dW[d] + jnp.sqrt(1 - rho ** 2) * dZ[d]
            v = v - kappa * v * DT + nu * jnp.sqrt(DT) * dB
            xs.append(x)
        return jnp.stack(xs, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="bench_data/D2/test.npz")
    ap.add_argument("--n_surfaces", type=int, default=8)
    ap.add_argument("--ensemble", type=int, default=4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--paths", type=int, default=4096, help="calibration paths (fixed noise)")
    ap.add_argument("--price_paths", type=int, default=50000, help="fresh paths for exotic pricing")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--pred_dir", default="results/trackB/preds")
    ap.add_argument("--out", default="results/trackB")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    te = load_D2(args.test)
    taus_d, xg, taus = te["taus_days"], te["xgrid"], te["taus"]
    S_all = te["iv"].shape[0]
    # pick a mix of GARCH and Heston surfaces
    gi = np.nonzero(te["family"] == "garch")[0]; hi = np.nonzero(te["family"] == "heston")[0]
    nh = min(len(hi), max(1, args.n_surfaces // 4))
    sel = np.sort(np.r_[gi[:args.n_surfaces - nh], hi[:nh]])
    model = NSDE(args.width)
    K = jnp.asarray(np.exp(xg), jnp.float32)
    isput = jnp.asarray(xg < 0)

    ivs, ex_mean, ex_lo, ex_hi, hes_ex, hes_iv = [], [], [], [], [], []
    for s in sel:
        clock = Timer()
        m = te["mask_full"][s]; q = te["quote_full"][s]
        qn = np.nan_to_num(q, nan=0.2)
        price = jnp.asarray(bs_otm(qn, xg, taus[:, None]), jnp.float32)
        vega = jnp.asarray(np.maximum(bs_vega(qn, xg, taus[:, None]), 1e-6), jnp.float32)
        mk = jnp.asarray(m.astype(np.float32))
        s0 = float(np.nanmedian(q))

        def calib_prices(p, dW, dZ):
            lnS = model.apply({"params": p}, s0, dW, dZ)
            ST = jnp.exp(lnS[:, taus_d])  # [N, m]
            pay = jnp.where(isput[None], jnp.maximum(K[None] - ST[..., None], 0), jnp.maximum(ST[..., None] - K[None], 0))
            return pay.mean(0)  # [m, n]

        def loss(p, dW, dZ):
            r = (calib_prices(p, dW, dZ) - price) / vega * 100  # ~ IV error in vol points
            return jnp.sum(mk * r ** 2) / mk.sum()

        opt = optax.adam(optax.cosine_decay_schedule(args.lr, args.steps, 0.1))
        keys = jax.random.split(jax.random.PRNGKey(args.seed + int(s)), args.ensemble)
        params = jax.vmap(lambda k: model.init(k, s0, jnp.zeros((DAYS, 2)), jnp.zeros((DAYS, 2)))["params"])(keys)
        st = jax.vmap(opt.init)(params)
        kn = jax.random.split(jax.random.PRNGKey(1000 + int(s)), 2)
        dW = jax.random.normal(kn[0], (DAYS, args.paths)); dZ = jax.random.normal(kn[1], (DAYS, args.paths))

        @jax.jit
        def step(params, st):
            l, g = jax.vmap(jax.value_and_grad(loss), in_axes=(0, None, None))(params, dW, dZ)
            u, st = jax.vmap(opt.update)(g, st, params)
            return jax.vmap(optax.apply_updates)(params, u), st, l

        for it in range(args.steps):
            params, st, l = step(params, st)
        # price exotics with fresh noise per member
        sim = jax.jit(lambda p, a, b: model.apply({"params": p}, s0, a, b))
        exs, cps = [], []
        for e in range(args.ensemble):
            pe = jax.tree_util.tree_map(lambda x: x[e], params)
            k1, k2 = jax.random.split(jax.random.PRNGKey(5000 + 17 * int(s) + e))
            lnS = np.asarray(sim(pe, jax.random.normal(k1, (DAYS, args.price_paths)), jax.random.normal(k2, (DAYS, args.price_paths))))
            ex = exotic_prices(lnS)
            exs.append([ex[k] for k in EXOTICS])
            ST = np.exp(lnS[:, taus_d])
            cps.append(np.where(xg[None] < 0, np.maximum(np.exp(xg)[None] - ST[..., None], 0),
                                np.maximum(ST[..., None] - np.exp(xg)[None], 0)).mean(0))
        exs = np.array(exs)
        ex_mean.append(exs.mean(0)); ex_lo.append(exs.min(0)); ex_hi.append(exs.max(0))
        ivs.append(implied_vol(np.mean(cps, 0), xg, taus[:, None]))
        # Heston baseline on the same quotes
        hiv, hp = fit_heston(te, "full", s)
        hes_iv.append(hiv)
        hx = exotic_prices(heston_paths(hp, DAYS, args.price_paths, np.random.default_rng(int(s))))
        hes_ex.append([hx[k] for k in EXOTICS])
        tex = te["exotics"][s]
        print(f"  surface {s} ({te['family'][s]}): calib RMSE ~{np.sqrt(np.asarray(l)).round(2)} vp; {clock():.0f}s\n"
              f"     truth  {np.round(tex, 4)}\n     nsde   {np.round(exs.mean(0), 4)}  band [{np.round(exs.min(0), 4)} , {np.round(exs.max(0), 4)}]\n"
              f"     heston {np.round(hes_ex[-1], 4)}", flush=True)

    p1 = os.path.join(args.pred_dir, "neural_sde_full.npz")
    save_pred(p1, "neural_sde", "full", np.array(ivs), exotics=np.array(ex_mean), exotics_lo=np.array(ex_lo),
              exotics_hi=np.array(ex_hi), surfaces=sel)
    p2 = os.path.join(args.pred_dir, "heston_calib_full.npz")
    save_pred(p2, "heston_calib", "full", np.array(hes_iv), exotics=np.array(hes_ex), surfaces=sel)
    for p in (p1, p2):
        score_and_save(args.test, p, args.out)


if __name__ == "__main__":
    main()
