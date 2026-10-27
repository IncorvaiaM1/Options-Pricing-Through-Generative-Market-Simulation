"""A7: conditional normalizing flow over the whole H-day future path (the "normalizing flow" from the pitch).

RealNVP-style affine coupling layers on y = asinh(x) of the normalized future returns (a bijection that tames
fat tails), conditioned on context features, with fixed random permutations between layers. Exact likelihood
(so it can be compared with RNADE / token AR on held-out NLL) and one-shot sampling (fast).
--base t uses a Student-t base distribution instead of a Gaussian (hypothesis H6 ablation: does a heavy-tailed
base fix the wings?). With the t base the asinh transform is switched off: a heavy-tailed base pushed through sinh
gives exponentially heavy tails, which blew up in testing. The t base supplies the tails on its own.

  python -m starters.a7_path_flow --data bench_data/A --out runs/a7
  python -m bench.trackA score --data bench_data/A --samples runs/a7/samples.npy
"""
import argparse

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.special import gammaln

from starters.common import add_also_eval_arg, also_eval, MLP, Timer, ctx_features, load_trackA, n_params, write_samples


class Coupling(nn.Module):
    H: int
    width: int

    @nn.compact
    def __call__(self, y, c, mask, inverse=False):
        h = MLP((self.width, self.width, 2 * self.H))(jnp.concatenate([y * mask, c], -1))
        s, t = jnp.split(h, 2, -1)
        s = 2.0 * jnp.tanh(s / 2.0) * (1 - mask)
        t = t * (1 - mask)
        if not inverse:
            return y * jnp.exp(s) + t, s.sum(-1)
        return (y - t) * jnp.exp(-s), -s.sum(-1)


class Flow(nn.Module):
    H: int
    n_layers: int = 8
    width: int = 256
    cond: int = 128

    def setup(self):
        self.enc = MLP((self.cond, self.cond))
        self.layers = [Coupling(self.H, self.width) for _ in range(self.n_layers)]
        rng = np.random.default_rng(0)
        self.perms = [rng.permutation(self.H) for _ in range(self.n_layers)]
        self.masks = [jnp.asarray((np.arange(self.H) % 2 == (i % 2)).astype(np.float32)) for i in range(self.n_layers)]

    def forward(self, y, ctx):
        """data -> base; returns z and log|det dz/dy|."""
        c = self.enc(ctx_features(ctx))
        ld = 0.0
        for l, p, m in zip(self.layers, self.perms, self.masks):
            y = y[:, p]
            y, d = l(y, c, m)
            ld = ld + d
        return y, ld

    def inverse(self, z, ctx):
        c = self.enc(ctx_features(ctx))
        for l, p, m in reversed(list(zip(self.layers, self.perms, self.masks))):
            z, _ = l(z, c, m, inverse=True)
            z = z[:, np.argsort(p)]
        return z

    def __call__(self, y, ctx):
        return self.forward(y, ctx)


def base_logp(z, base, nu):
    if base == "gauss":
        return jnp.sum(-0.5 * z ** 2 - 0.5 * jnp.log(2 * jnp.pi), -1)
    return jnp.sum(gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * jnp.log(nu * jnp.pi) - (nu + 1) / 2 * jnp.log1p(z ** 2 / nu), -1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench_data/A")
    ap.add_argument("--out", default="runs/a7")
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--base", default="gauss", choices=["gauss", "t"])
    ap.add_argument("--nu", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n_paths", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    add_also_eval_arg(ap)
    args = ap.parse_args(argv)

    cfg, train, val, ctx_eval = load_trackA(args.data)
    L, H = cfg["ctx_len"], cfg["horizon"]
    model = Flow(H, args.n_layers, args.width)
    key = jax.random.PRNGKey(args.seed)
    params = model.init(key, jnp.zeros((2, H)), jnp.zeros((2, L)))["params"]
    print(f"A7 path flow ({args.base} base): {n_params(params):,} params", flush=True)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.warmup_cosine_decay_schedule(
        0, args.lr, min(500, args.steps // 10 + 1), args.steps, args.lr * 0.05), weight_decay=1e-4))
    st = opt.init(params)

    use_asinh = args.base == "gauss"

    def nll(p, b):
        x = b[:, L:]
        y = jnp.arcsinh(x) if use_asinh else x
        z, ld = model.apply({"params": p}, y, b[:, :L])
        # log p(x) = log p_base(z) + log|dz/dy| + log|dy/dx|,  dy/dx = 1/sqrt(1+x^2) for asinh
        lp = base_logp(z, args.base, args.nu) + ld
        if use_asinh:
            lp = lp - 0.5 * jnp.sum(jnp.log1p(x ** 2), -1)
        return -jnp.mean(lp) / H  # nats per day (normalized units)

    @jax.jit
    def step(p, st, b):
        l, g = jax.value_and_grad(nll)(p, b)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    tr = jnp.asarray(train); va = jnp.asarray(val[:2048])
    val_nll = jax.jit(nll)
    clock = Timer()
    for i in range(1, args.steps + 1):
        key, kb = jax.random.split(key)
        params, st, l = step(params, st, tr[jax.random.randint(kb, (args.batch,), 0, tr.shape[0])])
        if i % args.log_every == 0 or i == 1:
            print(f"step {i:6d} train NLL/day {float(l):.4f}  val NLL/day {float(val_nll(params, va)):.4f}  {clock():.0f}s", flush=True)
    train_s = clock()
    final_val = float(val_nll(params, va))

    @jax.jit
    def sample(ctx, k):
        if args.base == "gauss":
            z = jax.random.normal(k, (ctx.shape[0], H))
        else:
            z = jax.random.t(k, args.nu, (ctx.shape[0], H))
        y = model.apply({"params": params}, z, ctx, method=Flow.inverse)
        return jnp.sinh(jnp.clip(y, -12, 12)) if use_asinh else y

    rep = np.repeat(ctx_eval, args.n_paths, axis=0)
    sample(jnp.asarray(rep[:2]), key)
    clock = Timer()
    outs = []
    for s in range(0, rep.shape[0], args.chunk):
        key, k = jax.random.split(key)
        outs.append(np.asarray(sample(jnp.asarray(rep[s:s + args.chunk]), k)))
    gen_s = clock()
    fut = np.concatenate(outs).reshape(ctx_eval.shape[0], args.n_paths, H) * cfg["scale"]
    meta = dict(label=f"a7_path_flow_{args.base}", n_params=n_params(params), train_seconds=train_s,
                                      gen_seconds=gen_s, val_nll_per_day_normalized=final_val)
    write_samples(args.out, fut, meta)
    also_eval(args, cfg, sample, key, meta)


if __name__ == "__main__":
    main()
