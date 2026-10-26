"""B3: attentive conditional neural process for vol-surface completion with uncertainty.

Context = observed quotes (k, log tau, IV); targets = every grid point. Each target attends to the context
points (multi-head cross-attention, a permutation-invariant set function: the geometric-DL tie-in) and also
gets a mean-pooled DeepSets summary. The decoder outputs a Gaussian (mean, sd) per target IV, trained by
Gaussian NLL on randomly masked training surfaces (both D2 families).
Hypothesis H9: lower held-out IV RMSE than SVI/spline/GP when quotes are sparse or wings are missing, AND
calibrated 90% bands (coverage near 0.90).

This is the deterministic-path (CNP) version; adding a latent path (full ANP) is the natural next step if
the bands come out over-confident on unusual surfaces.

  python -m starters.b3_neural_process --train bench_data/D2/train.npz --test bench_data/D2/test.npz
"""
import argparse
import os

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bench.surfaces import make_quotes
from bench.trackB import save_pred, score_and_save
from starters.common import MLP, Timer, load_D2, n_params


def coords(d):
    lt = np.log(d["taus"])
    K, T = np.meshgrid(d["ks"] / 3, (lt - lt.mean()) / lt.std())
    return np.stack([K, T], -1).reshape(-1, 2).astype(np.float32)  # [m*n, 2]


class ANP(nn.Module):
    width: int = 128
    heads: int = 4

    @nn.compact
    def __call__(self, xy, ctx_iv, ctx_mask):
        """xy [P,2] grid coords; ctx_iv [B,P] scaled IV (0 where unobserved); ctx_mask [B,P] -> mean, sd [B,P]."""
        B, P = ctx_iv.shape
        xyb = jnp.broadcast_to(xy[None], (B, P, 2))
        val = MLP((self.width, self.width))(jnp.concatenate([xyb, ctx_iv[..., None]], -1))
        pooled = (val * ctx_mask[..., None]).sum(1) / jnp.maximum(ctx_mask.sum(1, keepdims=True), 1.0)
        q = MLP((self.width, self.width))(xyb)
        k = MLP((self.width, self.width))(xyb)
        attn_mask = jnp.broadcast_to(ctx_mask[:, None, None, :] > 0, (B, self.heads, P, P))
        att = nn.MultiHeadDotProductAttention(num_heads=self.heads)(q, k, val, mask=attn_mask)
        h = jnp.concatenate([xyb, att, jnp.broadcast_to(pooled[:, None], (B, P, self.width))], -1)
        out = MLP((self.width, self.width, 2))(h)
        mean = out[..., 0]
        sd = 0.01 + nn.softplus(out[..., 1])  # in scaled units (0.1 = 1 vol pt * 10); floor = 0.1 vol pt
        return mean, sd


def scale_iv(iv):
    return (iv - 0.2) / 0.1


def unscale(x):
    return x * 0.1 + 0.2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="bench_data/D2/train.npz")
    ap.add_argument("--test", default="bench_data/D2/test.npz")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sets_per_surface", type=int, default=16)
    ap.add_argument("--scenarios", nargs="+", default=["per_mat5", "sparse10", "nowings"])
    ap.add_argument("--pred_dir", default="results/trackB/preds")
    ap.add_argument("--out", default="results/trackB")
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    tr, te = load_D2(args.train), load_D2(args.test)
    rng = np.random.default_rng(args.seed)
    S = tr["iv"].shape[0]
    M, Q, idx = [], [], []
    for s in range(S):
        for _ in range(args.sets_per_surface):
            sc = rng.choice(args.scenarios + ["random"])
            if sc == "random":  # random count 3..30 for variety
                iv = tr["iv"][s]
                mk = np.zeros(iv.size, bool); mk[rng.choice(iv.size, rng.integers(3, 31), replace=False)] = True
                mk = mk.reshape(iv.shape) & np.isfinite(iv)
                q = np.where(mk, iv + 0.005 * (1 + np.abs(tr["ks"]) / 2) * rng.standard_normal(iv.shape), np.nan)
                m_, q_ = mk, q
            else:
                m_, q_ = make_quotes(np.nan_to_num(tr["iv"][s], nan=0.2), sc, rng)
                m_ &= np.isfinite(tr["iv"][s])
            M.append(m_.reshape(-1)); Q.append(np.nan_to_num(q_, nan=0.2).reshape(-1)); idx.append(s)
    M = np.array(M, np.float32); Q = scale_iv(np.array(Q, np.float32)); idx = np.array(idx)
    Y = scale_iv(tr["iv"].reshape(S, -1)).astype(np.float32)
    YM = np.isfinite(Y).astype(np.float32); Y = np.nan_to_num(Y)
    xy = jnp.asarray(coords(tr))

    model = ANP(args.width)
    params = model.init(jax.random.PRNGKey(args.seed), xy, jnp.zeros((2, xy.shape[0])), jnp.ones((2, xy.shape[0])))["params"]
    print(f"B3 neural process: {n_params(params):,} params; {len(idx)} training quote sets", flush=True)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.cosine_decay_schedule(args.lr, args.steps, 0.05)))
    st = opt.init(params)

    def loss_fn(p, q, m, y, ym):
        mu, sd = model.apply({"params": p}, xy, q * m, m)
        nll = 0.5 * ((y - mu) / sd) ** 2 + jnp.log(sd)
        return jnp.sum(nll * ym) / jnp.sum(ym)

    @jax.jit
    def step(p, st, *b):
        l, g = jax.value_and_grad(loss_fn)(p, *b)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    clock = Timer()
    for it in range(1, args.steps + 1):
        b = rng.integers(0, len(idx), args.batch)
        params, st, l = step(params, st, jnp.asarray(Q[b]), jnp.asarray(M[b]), jnp.asarray(Y[idx[b]]), jnp.asarray(YM[idx[b]]))
        if it % args.log_every == 0 or it == 1:
            print(f"step {it:6d} NLL {float(l):.4f}  {clock():.0f}s", flush=True)

    pred = jax.jit(lambda q, m: model.apply({"params": params}, xy, q * m, m))
    shape = te["iv"].shape
    for sc in args.scenarios:
        m = te[f"mask_{sc}"].reshape(shape[0], -1).astype(np.float32)
        q = scale_iv(np.nan_to_num(te[f"quote_{sc}"], nan=0.2).reshape(shape[0], -1)).astype(np.float32)
        mu, sd = map(np.asarray, pred(jnp.asarray(q), jnp.asarray(m)))
        iv = unscale(mu).reshape(shape)
        lo = unscale(mu - 1.645 * sd).reshape(shape)
        hi = unscale(mu + 1.645 * sd).reshape(shape)
        path = os.path.join(args.pred_dir, f"neural_process_{sc}.npz")
        save_pred(path, "neural_process", sc, iv, iv_lo=lo, iv_hi=hi)
        score_and_save(args.test, path, args.out)


if __name__ == "__main__":
    main()
