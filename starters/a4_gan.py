"""A4: conditional GAN over the H-day future (placeholder until the teammate's GAN is plugged in).

Generator: MLP(context features, noise) -> H daily returns, output passed through sinh so fat tails are reachable.
Critic: the same GRU critic as A5 (context-conditioned), WGAN-GP loss. Kept deliberately simple, in the spirit of
Quant GANs / conditional time-series GANs, so H3 (GANs under-cover the tails) can be tested on the shared harness.
When the teammate's GAN is ready, have it write samples.npy for bench_data/A/eval_ctx.npy and score it the same way.

  python -m starters.a4_gan --data bench_data/A --out runs/a4
"""
import argparse

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from starters.a5_sde_gan import Critic
from starters.common import add_also_eval_arg, also_eval, MLP, Timer, ctx_features, load_trackA, n_params, write_samples


class Gen(nn.Module):
    H: int
    width: int = 256
    noise: int = 32

    @nn.compact
    def __call__(self, ctx, key):
        c = MLP((self.width, self.width))(ctx_features(ctx))
        z = jax.random.normal(key, (ctx.shape[0], self.noise))
        h = MLP((self.width, self.width, self.width, self.H))(jnp.concatenate([c, z], -1))
        return jnp.sinh(jnp.clip(h, -6, 6))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench_data/A")
    ap.add_argument("--out", default="runs/a4")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--n_critic", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr_g", type=float, default=1e-4)
    ap.add_argument("--lr_d", type=float, default=3e-4)
    ap.add_argument("--gp", type=float, default=10.0)
    ap.add_argument("--n_paths", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    add_also_eval_arg(ap)
    args = ap.parse_args(argv)
    cfg, train, val, ctx_eval = load_trackA(args.data)
    L, H = cfg["ctx_len"], cfg["horizon"]
    G, D = Gen(H, args.width), Critic(args.width // 2)
    key = jax.random.PRNGKey(args.seed)
    key, kg, kd = jax.random.split(key, 3)
    pg = G.init(kg, jnp.zeros((2, L)), kg)["params"]
    pd = D.init(kd, jnp.zeros((2, L)), jnp.zeros((2, H)))["params"]
    print(f"A4 GAN: G {n_params(pg):,}, D {n_params(pd):,} params", flush=True)
    og, od = optax.adam(args.lr_g, b1=0.0, b2=0.9), optax.adam(args.lr_d, b1=0.0, b2=0.9)
    sg, sd = og.init(pg), od.init(pd)

    def d_loss(pd, pg, ctx, real, k):
        k1, k2 = jax.random.split(k)
        fake = jax.lax.stop_gradient(G.apply({"params": pg}, ctx, k1))
        eps = jax.random.uniform(k2, (real.shape[0], 1))
        mix = eps * real + (1 - eps) * fake
        grad = jax.vmap(jax.grad(lambda x, c: D.apply({"params": pd}, c[None], x[None])[0]))(mix, ctx)
        gp = jnp.mean((jnp.linalg.norm(grad, axis=-1) - 1) ** 2)
        wd = D.apply({"params": pd}, ctx, real).mean() - D.apply({"params": pd}, ctx, fake).mean()
        return -wd + args.gp * gp, wd

    @jax.jit
    def d_step(pd, sd, pg, ctx, real, k):
        (_, wd), g = jax.value_and_grad(d_loss, has_aux=True)(pd, pg, ctx, real, k)
        u, sd = od.update(g, sd, pd)
        return optax.apply_updates(pd, u), sd, wd

    @jax.jit
    def g_step(pg, sg, pd, ctx, k):
        l, g = jax.value_and_grad(lambda p: -D.apply({"params": pd}, ctx, G.apply({"params": p}, ctx, k)).mean())(pg)
        u, sg = og.update(g, sg, pg)
        return optax.apply_updates(pg, u), sg, l

    tr = jnp.asarray(train)
    clock = Timer()
    for i in range(1, args.steps + 1):
        for _ in range(args.n_critic):
            key, kb, kk = jax.random.split(key, 3)
            b = tr[jax.random.randint(kb, (args.batch,), 0, tr.shape[0])]
            pd, sd, wd = d_step(pd, sd, pg, b[:, :L], b[:, L:], kk)
        key, kb, kk = jax.random.split(key, 3)
        b = tr[jax.random.randint(kb, (args.batch,), 0, tr.shape[0])]
        pg, sg, _ = g_step(pg, sg, pd, b[:, :L], kk)
        if i % args.log_every == 0 or i == 1:
            key, kk = jax.random.split(key)
            vb = jnp.asarray(val[:512]); fk = G.apply({"params": pg}, vb[:, :L], kk)
            print(f"step {i:6d}  W-dist est {float(wd):.4f}  gen std {float(fk.std()):.3f} (real {float(vb[:, L:].std()):.3f}) {clock():.0f}s", flush=True)
    train_s = clock()
    gen = jax.jit(lambda c, k: G.apply({"params": pg}, c, k))
    rep = np.repeat(ctx_eval, args.n_paths, axis=0)
    gen(jnp.asarray(rep[:2]), key)
    clock = Timer(); outs = []
    for s in range(0, rep.shape[0], args.chunk):
        key, k = jax.random.split(key)
        outs.append(np.asarray(gen(jnp.asarray(rep[s:s + args.chunk]), k)))
    fut = np.concatenate(outs).reshape(ctx_eval.shape[0], args.n_paths, H) * cfg["scale"]
    meta = dict(label="a4_gan", n_params=n_params(pg), train_seconds=train_s, gen_seconds=clock())
    write_samples(args.out, fut, meta)
    also_eval(args, cfg, gen, key, meta)


if __name__ == "__main__":
    main()
