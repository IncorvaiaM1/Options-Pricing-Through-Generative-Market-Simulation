"""A6: pure diffusion, no time structure.

One DDPM over the whole H-day future vector at once, conditioned on a summary of the context
(signed-log returns + realized-variance features). No autoregression: the model never sees its own
samples, so this isolates what A1's AR structure buys (hypothesis H5: similar vanilla prices,
worse path-dependent payoffs and vol clustering).

--transform asinh (default) trains on y = asinh(x) of the normalized returns and maps samples back with
sinh. It is a bijection, so no information is lost, but it tames the fat tails the denoiser sees; without
it the small-scale smoke test produced exploding samples. (Worth trying as an ablation on A1 too.)

Sampling clips the predicted clean sample x0_hat at each reverse step ("clip_denoised", standard in DDPM
code) to +-1.5x the largest |value| seen in training, in transformed space. With asinh that bound maps
back to hundreds of std, so it only removes numerical blowups, never real tail events.

  python -m bench.trackA export --out bench_data/A
  python -m starters.a6_pure_diffusion --data bench_data/A --out runs/a6
  python -m bench.trackA score --data bench_data/A --samples runs/a6/samples.npy
"""
import argparse

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ardiff.diffusion import make_schedule
from starters.common import (add_also_eval_arg, also_eval, AdaLNBlock, MLP, Timer, ctx_features, load_trackA, n_params, timestep_embedding,
                             write_samples)


class Denoiser(nn.Module):
    H: int
    width: int = 512
    depth: int = 4
    cond: int = 256

    @nn.compact
    def __call__(self, x_t, t, ctx):
        c = MLP((self.cond, self.cond))(ctx_features(ctx))
        c = c + MLP((self.cond, self.cond))(timestep_embedding(t, self.cond))
        h = nn.Dense(self.width)(x_t)
        for _ in range(self.depth):
            h = AdaLNBlock(self.width)(h, c)
        h = nn.LayerNorm(use_scale=False, use_bias=False)(h)
        return nn.Dense(self.H, kernel_init=nn.initializers.zeros)(h)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench_data/A")
    ap.add_argument("--out", default="runs/a6")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_paths", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--transform", default="asinh", choices=["asinh", "none"])
    ap.add_argument("--var_type", default="large", choices=["large", "small"], help="reverse-step variance; see ardiff/diffusion.py")
    ap.add_argument("--clip_mult", type=float, default=1.5, help="clip x0_hat at this x max|train| (<=0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    add_also_eval_arg(ap)
    args = ap.parse_args(argv)

    cfg, train, val, ctx_eval = load_trackA(args.data)
    L, H = cfg["ctx_len"], cfg["horizon"]
    sched = make_schedule(args.T, "linear")
    model = Denoiser(H, args.width, args.depth)
    key = jax.random.PRNGKey(args.seed)
    key, k0 = jax.random.split(key)
    params = model.init(k0, jnp.zeros((2, H)), jnp.zeros((2,), jnp.int32), jnp.zeros((2, L)))["params"]
    print(f"A6 pure diffusion: {n_params(params):,} params; train {train.shape}", flush=True)
    lr = optax.warmup_cosine_decay_schedule(0, args.lr, min(500, args.steps // 10 + 1), args.steps, args.lr * 0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=0.01))
    st = opt.init(params)

    def loss_fn(p, ctx, x0, k):
        k1, k2 = jax.random.split(k)
        t = jax.random.randint(k1, (x0.shape[0],), 0, args.T)
        eps = jax.random.normal(k2, x0.shape)
        ab = sched["abar"][t][:, None]
        xt = jnp.sqrt(ab) * x0 + jnp.sqrt(1 - ab) * eps
        return jnp.mean((model.apply({"params": p}, xt, t, ctx) - eps) ** 2)

    @jax.jit
    def step(p, st, ctx, x0, k):
        l, g = jax.value_and_grad(loss_fn)(p, ctx, x0, k)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    fwd = jnp.arcsinh if args.transform == "asinh" else (lambda x: x)
    inv = np.sinh if args.transform == "asinh" else (lambda x: x)
    tr = jnp.asarray(train); va = jnp.asarray(val)
    clock = Timer()
    for i in range(1, args.steps + 1):
        key, kb, ks = jax.random.split(key, 3)
        b = tr[jax.random.randint(kb, (args.batch,), 0, tr.shape[0])]
        params, st, l = step(params, st, b[:, :L], fwd(b[:, L:]), ks)
        if i % args.log_every == 0 or i == 1:
            key, kv, kl = jax.random.split(key, 3)
            vb = va[jax.random.randint(kv, (512,), 0, va.shape[0])]
            print(f"step {i:6d} train {float(l):.4f} val {float(loss_fn(params, vb[:, :L], fwd(vb[:, L:]), kl)):.4f} {clock():.0f}s", flush=True)
    train_s = clock()

    @jax.jit
    def sample(ctx, k):
        k, k0 = jax.random.split(k)
        x = jax.random.normal(k0, (ctx.shape[0], H))

        def body(carry, t):
            x, k = carry
            k, kn = jax.random.split(k)
            e = model.apply({"params": params}, x, jnp.full((x.shape[0],), t), ctx)
            a, b, ab = sched["alphas"][t], sched["betas"][t], sched["abar"][t]
            ab_prev = jnp.where(t > 0, sched["abar"][jnp.maximum(t - 1, 0)], 1.0)
            x0 = jnp.clip((x - jnp.sqrt(1 - ab) * e) / jnp.sqrt(ab), -clip, clip)
            mean = (jnp.sqrt(ab_prev) * b * x0 + jnp.sqrt(a) * (1 - ab_prev) * x) / (1 - ab)
            var = b if args.var_type == "large" else b * (1 - ab_prev) / (1 - ab)
            x = mean + jnp.where(t > 0, jnp.sqrt(var), 0.0) * jax.random.normal(kn, x.shape)
            return (x, k), None

        (x, _), _ = jax.lax.scan(body, (x, k), jnp.arange(args.T - 1, -1, -1))
        return x

    clip = float(args.clip_mult * jnp.abs(fwd(tr[:, L:])).max()) if args.clip_mult > 0 else 1e30
    rep = np.repeat(ctx_eval, args.n_paths, axis=0)
    sample(jnp.asarray(rep[:2]), key)
    clock = Timer()
    outs = []
    for s in range(0, rep.shape[0], args.chunk):
        key, k = jax.random.split(key)
        outs.append(np.asarray(sample(jnp.asarray(rep[s:s + args.chunk]), k)))
    gen_s = clock()
    fut = inv(np.concatenate(outs)).reshape(ctx_eval.shape[0], args.n_paths, H) * cfg["scale"]
    meta = dict(label="a6_pure_diffusion", n_params=n_params(params),
                                      train_seconds=train_s, gen_seconds=gen_s)
    write_samples(args.out, fut, meta)
    also_eval(args, cfg, lambda c, k: inv(np.asarray(sample(c, k))), key, meta)


if __name__ == "__main__":
    main()
