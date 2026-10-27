"""Train the AR backbone with a diffusion head (ours) or a token head (baseline).

  python -m ardiff.train --head diffusion --out runs/diff
  python -m ardiff.train --head token --n_bins 256 --bin_scheme quantile --out runs/tok
"""
import os as _os
import sys as _sys

if __package__ in (None, ""):  # also allow `python path/to/file.py`, not only `python -m pkg.file`
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ardiff import data as D
from ardiff import diffusion as dif
from ardiff import tokens as tok
from ardiff.model import ARModel


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="diffusion", choices=["diffusion", "token"])
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--data", default=None, help="train on an exported data folder (bench.trackA export or bench.real export) "
                                                  "instead of generating synthetic data here; then use ardiff.export_samples")
    # data
    ap.add_argument("--ctx_len", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--n_eval", type=int, default=32)
    ap.add_argument("--nu", type=float, default=5.0)
    ap.add_argument("--data_seed", type=int, default=0)
    # model
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--head_width", type=int, default=256)
    ap.add_argument("--head_depth", type=int, default=3)
    ap.add_argument("--diff_T", type=int, default=100)
    ap.add_argument("--diff_schedule", default="linear", choices=["linear", "cosine"])
    ap.add_argument("--diff_var", default="large", choices=["large", "small"], help="reverse-step variance (see diffusion.py)")
    ap.add_argument("--diff_mul", type=int, default=4)
    ap.add_argument("--n_bins", type=int, default=256)
    ap.add_argument("--bin_scheme", default="quantile", choices=["quantile", "uniform"])
    ap.add_argument("--bin_clip", type=float, default=6.0)
    # optim
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def build(cfg):
    model = ARModel(head=cfg["head"], d=cfg["d"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                    max_len=cfg["ctx_len"] + cfg["horizon"], dropout=cfg["dropout"],
                    head_width=cfg["head_width"], head_depth=cfg["head_depth"], n_bins=cfg["n_bins"])
    return model


def main(argv=None):
    args = get_args(argv)
    cfg = vars(args)
    os.makedirs(args.out, exist_ok=True)
    print("devices:", jax.devices(), flush=True)

    garch = D.GarchT(nu=args.nu)
    if args.data:
        dcfg = json.load(open(os.path.join(args.data, "config.json")))
        scale = dcfg["scale"]
        train = jnp.asarray(np.load(os.path.join(args.data, "train.npy")) / scale, jnp.float32)
        val = jnp.asarray(np.load(os.path.join(args.data, "val.npy")) / scale, jnp.float32)
        for k in ("ctx_len", "horizon", "n_train", "n_val", "n_eval", "nu", "data_seed"):
            cfg[k] = dcfg.get(k)  # record what the data really is
        cfg["data_kind"] = dcfg.get("kind", "synthetic")
    else:
        splits = D.make_splits(garch, args.ctx_len, args.horizon, args.n_train, args.n_val, args.n_eval, args.data_seed)
        scale = splits["scale"]
        train = jnp.asarray(splits["train"] / scale, jnp.float32)
        val = jnp.asarray(splits["val"] / scale, jnp.float32)
    print(f"train {train.shape}  scale(daily std)={scale:.5f}  max|x|={float(jnp.abs(train).max()):.1f} std units", flush=True)

    model = build(cfg)
    sched = dif.make_schedule(args.diff_T, args.diff_schedule)
    bins = tok.fit_bins(np.asarray(train), args.n_bins, args.bin_scheme, args.bin_clip) if args.head == "token" else None

    key = jax.random.PRNGKey(args.seed)
    key, kinit = jax.random.split(key)
    params = model.init(kinit, train[:2, :-1])["params"]
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"params: {n_params:,}", flush=True)

    warmup = min(args.warmup, max(1, args.steps // 10))
    lr = optax.warmup_cosine_decay_schedule(0.0, args.lr, warmup, args.steps, args.lr * 0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=args.wd))
    opt_state = opt.init(params)

    def loss_fn(params, xb, key, train_mode):
        kd, kl = jax.random.split(key)
        x_in, y = xb[:, :-1], xb[:, 1:]
        z = model.apply({"params": params}, x_in, train_mode, method=ARModel.encode, rngs={"dropout": kd})
        z = z.reshape(-1, z.shape[-1])
        y = y.reshape(-1)
        if args.head == "diffusion":
            eps_fn = lambda xt, t, zz: model.apply({"params": params}, xt, t, zz, method=ARModel.eps)
            return dif.diffusion_loss(eps_fn, z, y, kl, sched, args.diff_mul)
        logits = model.apply({"params": params}, z, method=ARModel.logits)
        return optax.softmax_cross_entropy_with_integer_labels(logits, tok.encode(y, bins)).mean()

    @jax.jit
    def train_step(params, opt_state, xb, key):
        loss, g = jax.value_and_grad(loss_fn)(params, xb, key, True)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    @jax.jit
    def val_loss(params, xb, key):
        return loss_fn(params, xb, key, False)

    hist = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        key, kb, ks = jax.random.split(key, 3)
        idx = jax.random.randint(kb, (args.batch,), 0, train.shape[0])
        params, opt_state, loss = train_step(params, opt_state, train[idx], ks)
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            key, kv1, kv2 = jax.random.split(key, 3)
            vidx = jax.random.randint(kv1, (min(args.batch, val.shape[0]),), 0, val.shape[0])  # same size as a training batch (memory)
            vl = float(val_loss(params, val[vidx], kv2))
            el = time.time() - t0
            print(f"step {step:6d}  train {float(loss):.4f}  val {vl:.4f}  {el:7.1f}s", flush=True)
            hist.append(dict(step=step, train=float(loss), val=vl, sec=el))

    ckpt = dict(cfg=cfg, params=jax.device_get(params), bins=None if bins is None else jax.device_get(bins),
                scale=scale, garch=garch.to_dict(), n_params=n_params, train_seconds=time.time() - t0, hist=hist)
    with open(os.path.join(args.out, "ckpt.pkl"), "wb") as f:
        pickle.dump(ckpt, f)
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(dict(cfg=cfg, n_params=n_params, hist=hist), f, indent=2)
    print(f"saved {args.out}/ckpt.pkl", flush=True)


if __name__ == "__main__":
    main()
