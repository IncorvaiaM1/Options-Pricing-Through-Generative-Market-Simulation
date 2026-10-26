"""B1: amortized energy-based model for risk-neutral densities from sparse quotes.

p(z | quotes, tau) proportional to exp(-E_theta(z; c, tau)) * N(z; 0, s_hat^2), on a fixed grid (1-D, so the normalizer
is exact by quadrature; no Langevin needed). c = DeepSets encoding of the observed quote set (permutation invariant).
Afterwards an exponential tilt exp(theta x) is solved by Newton so the forward is exactly 1 (martingale).

Classical max-entropy (Buchen & Kelly) is the special case E = sum_i lambda_i payoff_i(x) fitted per surface; here the
network is trained across many synthetic surfaces, so it learns what plausible densities look like
(hypothesis H8: lower Wasserstein-1 than classical max-ent / SVI with 5 to 10 quotes).

Loss on training surfaces (GARCH family, where true densities are known):
  cross-entropy to the true density  +  w_price * mean squared (price error / quote noise) on the observed quotes.

  python -m starters.b1_ebm --train bench_data/D2/train.npz --test bench_data/D2/test.npz
"""
import argparse
import os

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bench.finance import bs_otm, bs_vega
from bench.surfaces import SCENARIOS, make_quotes
from bench.trackB import save_pred, score_and_save
from starters.common import MLP, Timer, density_to_iv, load_D2, n_params

SIG_REF = 0.2


def point_features(d, mask, quotes):
    """-> [..., m*n, 4]: (k/3, standardized log tau, (iv - 0.2)/0.1 or 0, observed flag)."""
    ks, lt = d["ks"], np.log(d["taus"])
    K, T = np.meshgrid(ks / 3, (lt - lt.mean()) / lt.std())
    base = np.stack([K, T], -1).reshape(-1, 2)
    q = np.where(mask, (np.nan_to_num(quotes) - 0.2) / 0.1, 0.0).reshape(mask.shape[0], -1)
    m = mask.reshape(mask.shape[0], -1).astype(np.float32)
    return np.concatenate([np.broadcast_to(base, (mask.shape[0],) + base.shape), q[..., None], m[..., None]], -1).astype(np.float32)


def quote_targets(d, mask, quotes, noise_vp=0.5):
    x, taus = d["xgrid"], d["taus"][:, None]
    q = np.nan_to_num(quotes, nan=0.2)
    price = bs_otm(q, x, taus)
    nz = bs_vega(q, x, taus) * noise_vp / 100 * (1 + np.abs(d["ks"])[None, :] / 2)
    return price.astype(np.float32), np.maximum(nz, 1e-7).astype(np.float32)


def payoff_tensor(d):
    """[m, n, G] OTM payoff of each grid strike on each density grid point."""
    z = d["zgrid"]
    out = np.empty(d["xgrid"].shape + (len(z),))
    for j, t in enumerate(d["taus"]):
        S = np.exp(z * SIG_REF * np.sqrt(t))
        for i, kx in enumerate(d["xgrid"][j]):
            K = np.exp(kx)
            out[j, i] = np.maximum(K - S, 0) if kx < 0 else np.maximum(S - K, 0)
    return out.astype(np.float32)


class EBM(nn.Module):
    m: int
    G: int
    width: int = 256

    @nn.compact
    def __call__(self, feats, zgrid, lt_std, s_hat):
        obs = feats[..., 3:4]
        h = MLP((self.width, self.width))(feats)
        pooled = (h * obs).sum(-2) / jnp.maximum(obs.sum(-2), 1.0)
        c = MLP((self.width, self.width))(jnp.concatenate([pooled, (h * obs).max(-2)], -1))  # [B, W]
        B = feats.shape[0]
        zz = jnp.broadcast_to(zgrid[None, None, :, None], (B, self.m, self.G, 1))
        tt = jnp.broadcast_to(lt_std[None, :, None, None], (B, self.m, self.G, 1))
        cc = jnp.broadcast_to(c[:, None, None, :], (B, self.m, self.G, c.shape[-1]))
        zs = zz / s_hat[:, None, None, None]
        E = MLP((self.width, self.width, 1))(jnp.concatenate([zs, zs ** 2 / 10, tt, cc], -1))[..., 0]
        return -E - 0.5 * zs[..., 0] ** 2  # logits [B, m, G]


def tilt_to_forward(logits, x, iters=8):
    """Exponential tilt so that sum p e^x = 1 per (b, j). x: [m, G]."""
    th = jnp.zeros(logits.shape[:-1])
    for _ in range(iters):
        l = logits + th[..., None] * x
        p = jax.nn.softmax(l, -1)
        m1 = (p * jnp.exp(x)).sum(-1)
        m2 = (p * jnp.exp(x) * x).sum(-1)
        mx = (p * x).sum(-1)
        # d/dth log E_p_th[e^x] = E_tilted[x] ; Newton on f(th) = log m1 = 0
        th = th - jnp.log(m1) / (m2 / m1 - mx + 1e-9)
    return logits + th[..., None] * x


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="bench_data/D2/train.npz")
    ap.add_argument("--test", default="bench_data/D2/test.npz")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w_price", type=float, default=0.1)
    ap.add_argument("--sets_per_surface", type=int, default=8)
    ap.add_argument("--scenarios", nargs="+", default=["per_mat5", "sparse10", "nowings"])
    ap.add_argument("--pred_dir", default="results/trackB/preds")
    ap.add_argument("--out", default="results/trackB")
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    tr, te = load_D2(args.train), load_D2(args.test)
    g = tr["family"] == "garch"
    dens_tr = tr["density"][g].astype(np.float32)
    iv_tr = tr["iv"][g]
    rng = np.random.default_rng(args.seed)
    # pool of random quote sets per training surface (mixing the three sparse scenarios)
    M, Q, idx = [], [], []
    for s in range(iv_tr.shape[0]):
        for _ in range(args.sets_per_surface):
            m_, q_ = make_quotes(np.nan_to_num(iv_tr[s], nan=0.2), rng.choice(args.scenarios), rng)
            M.append(m_); Q.append(q_); idx.append(s)
    M, Q, idx = np.array(M), np.array(Q), np.array(idx)
    F = point_features(tr, M, Q)
    P, NZ = quote_targets(tr, M, Q)
    SH = np.array([np.median(q[m]) for q, m in zip(Q, M)], np.float32) / SIG_REF
    PAY = jnp.asarray(payoff_tensor(tr))
    z = jnp.asarray(tr["zgrid"], jnp.float32)
    dz = float(z[1] - z[0])
    X = jnp.asarray(tr["zgrid"][None, :] * SIG_REF * np.sqrt(tr["taus"])[:, None], jnp.float32)
    lt = np.log(tr["taus"]); LT = jnp.asarray((lt - lt.mean()) / lt.std(), jnp.float32)
    m_, G = len(tr["taus"]), len(tr["zgrid"])
    print(f"B1 EBM: {g.sum()} GARCH training surfaces, {len(idx)} quote sets", flush=True)

    model = EBM(m_, G, args.width)
    key = jax.random.PRNGKey(args.seed)
    params = model.init(key, jnp.asarray(F[:2]), z, LT, jnp.asarray(SH[:2]))["params"]
    print(f"params {n_params(params):,}", flush=True)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.cosine_decay_schedule(args.lr, args.steps, 0.05)))
    st = opt.init(params)

    def forward(p, f, sh):
        return tilt_to_forward(model.apply({"params": p}, f, z, LT, sh), X)

    def loss_fn(p, f, sh, dtrue, mask, price, nz):
        logits = forward(p, f, sh)
        logp = jax.nn.log_softmax(logits, -1) - jnp.log(dz)
        ce = -(dtrue * dz * logp).sum(-1).mean()
        pm = jnp.einsum("bjg,jig->bji", jax.nn.softmax(logits, -1), PAY)
        pe = jnp.sum(mask * ((pm - price) / nz) ** 2) / jnp.maximum(mask.sum(), 1)
        return ce + args.w_price * pe, (ce, pe)

    @jax.jit
    def step(p, st, *b):
        (l, aux), gr = jax.value_and_grad(loss_fn, has_aux=True)(p, *b)
        u, st = opt.update(gr, st, p)
        return optax.apply_updates(p, u), st, l, aux

    clock = Timer()
    for it in range(1, args.steps + 1):
        b = rng.integers(0, len(idx), args.batch)
        params, st, l, (ce, pe) = step(params, st, jnp.asarray(F[b]), jnp.asarray(SH[b]), jnp.asarray(dens_tr[idx[b]]),
                                       jnp.asarray(M[b].astype(np.float32)), jnp.asarray(P[b]), jnp.asarray(NZ[b]))
        if it % args.log_every == 0 or it == 1:
            print(f"step {it:6d} loss {float(l):.4f}  CE {float(ce):.4f}  price chi2 {float(pe):.3f}  {clock():.0f}s", flush=True)

    fwd = jax.jit(forward)
    zg = te["zgrid"]
    for sc in args.scenarios:
        Mt, Qt = te[f"mask_{sc}"], te[f"quote_{sc}"]
        Ft = point_features(te, Mt, Qt)
        SHt = np.array([np.median(q[m]) for q, m in zip(Qt, Mt)], np.float32) / SIG_REF
        logits = np.asarray(fwd(params, jnp.asarray(Ft), jnp.asarray(SHt)))
        dens = np.exp(logits - logits.max(-1, keepdims=True))
        dens = dens / (dens.sum(-1, keepdims=True) * (zg[1] - zg[0]))
        iv = density_to_iv(dens, zg, te["xgrid"], te["taus"])
        path = os.path.join(args.pred_dir, f"ebm_{sc}.npz")
        save_pred(path, "ebm", sc, iv, density=dens)
        score_and_save(args.test, path, args.out)


if __name__ == "__main__":
    main()
