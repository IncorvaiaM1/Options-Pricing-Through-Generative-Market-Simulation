"""A5: neural SDE trained as a GAN (after Kidger et al., ICML 2021), conditioned on the context.

Generator (continuous time, Euler-Maruyama with `--substeps` per day):
    c   = MLP(context features)
    h_0 = MLP([c, v]),  v ~ N(0, I)
    dh  = f(t, h, c) dt + g(t, h, c) dW_t          f: R^dh, g: R^{dh x dw}, both tanh-bounded (Lipschitz-ish)
    X_t = readout(h_t) - readout(h_0)               cumulative log price in normalized units
    daily returns r_d = X_d - X_{d-1}
Critic: GRU over (r_d, d/H) with initial state from the context; WGAN-GP loss.
Kidger et al. use a neural CDE critic; a GRU is the discrete-time cousin and is easier to start with.
Swap in a neural CDE (e.g. with diffrax) as a follow-up.

Hypothesis H4: stable paths and good vol clustering, but Brownian driving noise underweights jumps, so the
far wings (|k| = 4) trail A1. Ablation: add a jump term (--jumps) with learned intensity and size scale.

  python -m starters.a5_sde_gan --data bench_data/A --out runs/a5
  python -m bench.trackA score --data bench_data/A --samples runs/a5/samples.npy
"""
import argparse

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from starters.common import add_also_eval_arg, also_eval, MLP, Timer, ctx_features, load_trackA, n_params, write_samples


class Generator(nn.Module):
    H: int
    dh: int = 32
    dw: int = 16
    width: int = 128
    substeps: int = 4
    jumps: bool = False

    @nn.compact
    def __call__(self, ctx, key):
        B = ctx.shape[0]
        c = MLP((self.width, self.width))(ctx_features(ctx))
        k0, kw, kj1, kj2 = jax.random.split(key, 4)
        h = MLP((self.width, self.dh))(jnp.concatenate([c, jax.random.normal(k0, (B, self.dh))], -1))
        drift = MLP((self.width, self.width, self.dh))
        diff = MLP((self.width, self.width, self.dh * self.dw))
        readout = nn.Dense(1)
        if self.jumps:
            jump_net = MLP((self.width, 2))
        n = self.H * self.substeps
        dt = 1.0 / self.substeps
        dW = jax.random.normal(kw, (n, B, self.dw)) * jnp.sqrt(dt)
        U = jax.random.uniform(kj1, (n, B))
        Z = jax.random.normal(kj2, (n, B))
        x0 = readout(h)[:, 0]
        xs = []
        for i in range(n):  # unrolled; H * substeps is small (80 by default)
            t = jnp.full((B, 1), i * dt / self.H)
            inp = jnp.concatenate([t, h, c], -1)
            f = jnp.tanh(drift(inp))
            g = jnp.tanh(diff(inp)).reshape(B, self.dh, self.dw)
            h = h + f * dt + jnp.einsum("bij,bj->bi", g, dW[i])
            if self.jumps:
                lam_size = jump_net(inp)
                lam = nn.sigmoid(lam_size[:, 0]) * 0.2 * dt          # jump prob per substep (<= 0.2 per day)
                size = nn.softplus(lam_size[:, 1]) * Z[i]             # jump size in normalized units
                h = h.at[:, 0].add(jnp.where(U[i] < lam, size, 0.0))  # jumps enter through the first hidden unit
            if (i + 1) % self.substeps == 0:
                xs.append(readout(h)[:, 0])
        X = jnp.stack([x0] + xs, 1)
        return jnp.diff(X, axis=1)  # [B, H] daily returns (normalized)


class Critic(nn.Module):
    width: int = 128

    @nn.compact
    def __call__(self, ctx, r):
        B, H = r.shape
        c = MLP((self.width, self.width))(ctx_features(ctx))
        h0 = jnp.tanh(nn.Dense(self.width)(c))
        seq = jnp.stack([jnp.sign(r) * jnp.log1p(jnp.abs(r)), jnp.broadcast_to(jnp.arange(H) / H, (B, H))], -1)
        rnn = nn.RNN(nn.GRUCell(features=self.width))
        hs = rnn(seq, initial_carry=h0)
        return MLP((self.width, 1))(jnp.concatenate([hs[:, -1], c], -1))[:, 0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench_data/A")
    ap.add_argument("--out", default="runs/a5")
    ap.add_argument("--dh", type=int, default=32)
    ap.add_argument("--dw", type=int, default=16)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--jumps", action="store_true")
    ap.add_argument("--steps", type=int, default=10000, help="generator steps")
    ap.add_argument("--n_critic", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr_g", type=float, default=1e-4)
    ap.add_argument("--lr_d", type=float, default=3e-4)
    ap.add_argument("--gp", type=float, default=10.0)
    ap.add_argument("--n_paths", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    add_also_eval_arg(ap)
    args = ap.parse_args(argv)

    cfg, train, val, ctx_eval = load_trackA(args.data)
    L, H = cfg["ctx_len"], cfg["horizon"]
    G = Generator(H, args.dh, args.dw, args.width, args.substeps, args.jumps)
    D = Critic(args.width)
    key = jax.random.PRNGKey(args.seed)
    key, kg, kd, ks = jax.random.split(key, 4)
    dummy = jnp.zeros((2, L))
    pg = G.init(kg, dummy, ks)["params"]
    pd = D.init(kd, dummy, jnp.zeros((2, H)))["params"]
    print(f"A5 SDE-GAN: G {n_params(pg):,} params, D {n_params(pd):,} params; jumps={args.jumps}", flush=True)
    og = optax.adam(args.lr_g, b1=0.0, b2=0.9)
    od = optax.adam(args.lr_d, b1=0.0, b2=0.9)
    sg, sd = og.init(pg), od.init(pd)

    def d_loss(pd, pg, ctx, real, k):
        k1, k2 = jax.random.split(k)
        fake = jax.lax.stop_gradient(G.apply({"params": pg}, ctx, k1))
        dr = D.apply({"params": pd}, ctx, real)
        df = D.apply({"params": pd}, ctx, fake)
        eps = jax.random.uniform(k2, (real.shape[0], 1))
        mix = eps * real + (1 - eps) * fake
        grad = jax.vmap(jax.grad(lambda x, c: D.apply({"params": pd}, c[None], x[None])[0]))(mix, ctx)
        gp = jnp.mean((jnp.linalg.norm(grad, axis=-1) - 1) ** 2)
        return df.mean() - dr.mean() + args.gp * gp, dr.mean() - df.mean()

    def g_loss(pg, pd, ctx, k):
        return -D.apply({"params": pd}, ctx, G.apply({"params": pg}, ctx, k)).mean()

    @jax.jit
    def d_step(pd, sd, pg, ctx, real, k):
        (l, wdist), g = jax.value_and_grad(d_loss, has_aux=True)(pd, pg, ctx, real, k)
        u, sd = od.update(g, sd, pd)
        return optax.apply_updates(pd, u), sd, wdist

    @jax.jit
    def g_step(pg, sg, pd, ctx, k):
        l, g = jax.value_and_grad(g_loss)(pg, pd, ctx, k)
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
        pg, sg, gl = g_step(pg, sg, pd, b[:, :L], kk)
        if i % args.log_every == 0 or i == 1:
            key, kk = jax.random.split(key)
            vb = jnp.asarray(val[:512])
            fk = G.apply({"params": pg}, vb[:, :L], kk)
            print(f"step {i:6d}  W-dist est {float(wd):.4f}  gen std {float(fk.std()):.3f} (real {float(vb[:, L:].std()):.3f})"
                  f"  gen max|r| {float(jnp.abs(fk).max()):.1f}  {clock():.0f}s", flush=True)
    train_s = clock()

    gen = jax.jit(lambda c, k: G.apply({"params": pg}, c, k))
    rep = np.repeat(ctx_eval, args.n_paths, axis=0)
    gen(jnp.asarray(rep[:2]), key)
    clock = Timer()
    outs = []
    for s in range(0, rep.shape[0], args.chunk):
        key, k = jax.random.split(key)
        outs.append(np.asarray(gen(jnp.asarray(rep[s:s + args.chunk]), k)))
    gen_s = clock()
    fut = np.concatenate(outs).reshape(ctx_eval.shape[0], args.n_paths, H) * cfg["scale"]
    meta = dict(label="a5_sde_gan" + ("_jumps" if args.jumps else ""),
                                      n_params=n_params(pg), train_seconds=train_s, gen_seconds=gen_s)
    write_samples(args.out, fut, meta)
    also_eval(args, cfg, gen, key, meta)


if __name__ == "__main__":
    main()
