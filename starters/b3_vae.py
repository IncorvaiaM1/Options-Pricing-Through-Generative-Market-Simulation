"""B3: VAE on vol surfaces (after Bergeron et al.; Ning et al.).

Trains a small VAE on full D2 training surfaces (13 strikes x 5 maturities), latent dim 2 to 4. Three outputs:
  1. factor check (H10): R^2 of level / skew / term-slope (and the true GARCH state vol) regressed on the
     latent means of the test surfaces, plus each latent's best-matching factor
  2. completion: for each test scenario, find the latent that best fits the noisy quotes
     (MAP: quote misfit / noise + ||z||^2), decode -> predicted surface, scored by bench.trackB
  3. stress walks: move each latent from -2.5 to 2.5 (others 0), decode, report the butterfly-arbitrage rate

  python -m starters.b3_vae --train bench_data/D2/train.npz --test bench_data/D2/test.npz --latent 3
"""
import argparse
import json
import os

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bench.trackB import butterfly_violation_rate, save_pred, score_and_save
from starters.common import MLP, Timer, load_D2, n_params


class VAE(nn.Module):
    P: int
    latent: int = 3
    width: int = 256

    def setup(self):
        self.enc = MLP((self.width, self.width, 2 * self.latent))
        self.dec = MLP((self.width, self.width, self.P))

    def encode(self, y):
        h = self.enc(y)
        return h[..., :self.latent], h[..., self.latent:]

    def decode(self, z):
        return self.dec(z)

    def __call__(self, y, key):
        mu, logv = self.encode(y)
        z = mu + jnp.exp(0.5 * logv) * jax.random.normal(key, mu.shape)
        return self.decode(z), mu, logv


def sc(iv):
    return (iv - 0.2) / 0.1


def un(x):
    return x * 0.1 + 0.2


def factors(iv, ks, fam, gparams):
    a = np.where(ks == 0)[0][0]
    m1, p1 = np.where(ks == -1)[0][0], np.where(ks == 1)[0][0]
    f = dict(level_atm20=iv[:, 2, a], skew20=iv[:, 2, m1] - iv[:, 2, p1], term_slope=iv[:, 4, a] - iv[:, 0, a])
    return f


def r2(y, Z):
    ok = np.isfinite(y)
    A = np.c_[Z[ok], np.ones(ok.sum())]
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    res = y[ok] - A @ coef
    return float(1 - res.var() / y[ok].var())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="bench_data/D2/train.npz")
    ap.add_argument("--test", default="bench_data/D2/test.npz")
    ap.add_argument("--latent", type=int, default=3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fit_steps", type=int, default=400)
    ap.add_argument("--scenarios", nargs="+", default=["per_mat5", "sparse10", "nowings"])
    ap.add_argument("--pred_dir", default="results/trackB/preds")
    ap.add_argument("--out", default="results/trackB")
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    tr, te = load_D2(args.train), load_D2(args.test)
    shape = te["iv"].shape[1:]
    P = int(np.prod(shape))

    def flat(iv):
        y = sc(iv.reshape(iv.shape[0], -1)).astype(np.float32)
        return np.nan_to_num(y), np.isfinite(y).astype(np.float32)

    Y, YM = flat(tr["iv"])
    model = VAE(P, args.latent, args.width)
    key = jax.random.PRNGKey(args.seed)
    params = model.init(key, jnp.zeros((2, P)), key)["params"]
    print(f"B3 VAE: latent {args.latent}, {n_params(params):,} params, {Y.shape[0]} training surfaces", flush=True)
    opt = optax.adam(optax.cosine_decay_schedule(args.lr, args.steps, 0.05))
    st = opt.init(params)
    log_sig = jnp.log(0.05)  # observation noise in scaled units (0.5 vol pt)

    def loss_fn(p, y, ym, k):
        rec, mu, logv = model.apply({"params": p}, y, k)
        nll = 0.5 * jnp.sum(ym * ((rec - y) / jnp.exp(log_sig)) ** 2, -1)
        kl = 0.5 * jnp.sum(jnp.exp(logv) + mu ** 2 - 1 - logv, -1)
        return jnp.mean(nll + args.beta * kl), (jnp.mean(nll), jnp.mean(kl))

    @jax.jit
    def step(p, st, y, ym, k):
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, y, ym, k)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l, aux

    rng = np.random.default_rng(args.seed)
    clock = Timer()
    for it in range(1, args.steps + 1):
        b = rng.integers(0, Y.shape[0], args.batch)
        key, k = jax.random.split(key)
        params, st, l, (nll, kl) = step(params, st, jnp.asarray(Y[b]), jnp.asarray(YM[b]), k)
        if it % args.log_every == 0 or it == 1:
            print(f"step {it:6d} loss {float(l):.2f}  recon {float(nll):.2f}  KL {float(kl):.2f}  {clock():.0f}s", flush=True)

    enc = jax.jit(lambda y: model.apply({"params": params}, y, method=VAE.encode)[0])
    dec = jax.jit(lambda z: model.apply({"params": params}, z, method=VAE.decode))

    # 1. factor check on the test set's true surfaces
    Yt, _ = flat(te["iv"])
    Z = np.asarray(enc(jnp.asarray(Yt)))
    F = factors(te["iv"], te["ks"], te["family"], te["garch_params"])
    g = te["family"] == "garch"
    if g.sum() > args.latent + 2:
        F["garch_state_vol"] = np.where(g, te["garch_params"][:, 5], np.nan)
    rep = dict(latent=args.latent, r2_factor_on_latents={k: r2(v, Z) for k, v in F.items()},
               best_factor_per_latent=[])
    for d in range(args.latent):
        cors = {k: float(np.corrcoef(Z[np.isfinite(v), d], v[np.isfinite(v)])[0, 1]) for k, v in F.items()}
        best = max(cors, key=lambda k: abs(cors[k]))
        rep["best_factor_per_latent"].append(dict(latent=d, factor=best, corr=cors[best]))
    # 3. stress walks
    walks = {}
    for d in range(args.latent):
        zz = np.zeros((11, args.latent), np.float32); zz[:, d] = np.linspace(-2.5, 2.5, 11)
        ivw = un(np.asarray(dec(jnp.asarray(zz)))).reshape((11,) + shape)
        walks[f"latent{d}"] = dict(butterfly_violation_rate=butterfly_violation_rate(ivw, te["xgrid"], te["taus"]),
                                   atm20_range_vp=[float(ivw[:, 2, 6].min() * 100), float(ivw[:, 2, 6].max() * 100)])
    rep["latent_walks"] = walks
    os.makedirs(args.out, exist_ok=True)
    json.dump(rep, open(os.path.join(args.out, f"vae_latent{args.latent}_factors.json"), "w"), indent=1)
    print("factor R^2 on latents:", {k: round(v, 3) for k, v in rep["r2_factor_on_latents"].items()})
    print("best factor per latent:", [(r["factor"], round(r["corr"], 2)) for r in rep["best_factor_per_latent"]])

    # 2. completion by MAP latent fit to quotes
    opt2 = optax.adam(5e-2)
    for scn in args.scenarios:
        m = te[f"mask_{scn}"].reshape(te["iv"].shape[0], -1).astype(np.float32)
        q = sc(np.nan_to_num(te[f"quote_{scn}"], nan=0.2).reshape(te["iv"].shape[0], -1)).astype(np.float32)
        nz = np.broadcast_to((0.05 * (1 + np.abs(te["ks"]) / 2))[None, :], shape).reshape(-1).astype(np.float32)
        z = jnp.zeros((m.shape[0], args.latent))
        s2 = opt2.init(z)

        def obj(z):
            r = (dec(z) - q) / nz
            return jnp.sum(m * r ** 2) + jnp.sum(z ** 2)

        g_ = jax.jit(jax.grad(obj))
        for _ in range(args.fit_steps):
            u, s2 = opt2.update(g_(z), s2, z)
            z = optax.apply_updates(z, u)
        iv = un(np.asarray(dec(z))).reshape(te["iv"].shape)
        path = os.path.join(args.pred_dir, f"vae{args.latent}_{scn}.npz")
        save_pred(path, f"vae{args.latent}", scn, iv)
        score_and_save(args.test, path, args.out)


if __name__ == "__main__":
    main()
