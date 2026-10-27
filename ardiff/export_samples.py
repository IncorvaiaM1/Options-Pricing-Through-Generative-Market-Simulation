"""Put an A1/A2 checkpoint on the shared Track A harness: writes samples.npy + meta.json.

  python -m ardiff.export_samples --ckpt runs/v0_diff/ckpt.pkl --data bench_data/A --n_paths 8192 --out runs/v0_diff
  python -m bench.trackA score --data bench_data/A --samples runs/v0_diff/samples.npy
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

from ardiff import tokens as tok
from ardiff.evaluate import generate, make_sampler
from ardiff.train import build


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="bench_data/A")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--n_paths", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--token_decode", default="mean", choices=["mean", "center", "dequant"])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    c = pickle.load(open(args.ckpt, "rb"))
    cfg = c["cfg"]
    dcfg = json.load(open(os.path.join(args.data, "config.json")))
    for k in ("ctx_len", "horizon"):
        assert cfg[k] == dcfg[k], f"checkpoint and exported data disagree on {k}"
    if dcfg.get("kind", "synthetic") == "synthetic":
        for k in ("n_train", "n_val", "n_eval", "nu", "data_seed"):
            if cfg.get(k) != dcfg.get(k):
                print(f"WARNING: checkpoint was trained with {k}={cfg.get(k)} but --data has {k}={dcfg.get(k)}")
    model = build(cfg)
    params = jax.tree_util.tree_map(jnp.asarray, c["params"])
    bins = None if c["bins"] is None else jax.tree_util.tree_map(jnp.asarray, c["bins"])
    if bins is not None and "means" not in bins:
        train = np.load(os.path.join(args.data, "train.npy")) / c["scale"]
        bins = tok.fit_bins(train, cfg["n_bins"], cfg["bin_scheme"], cfg["bin_clip"])
    sampler = make_sampler(model, params, cfg, bins, args.token_decode)
    ctx = np.load(os.path.join(args.data, "eval_ctx.npy")) / c["scale"]
    key = jax.random.PRNGKey(args.seed)
    generate(sampler, ctx[:1], 2, key, args.chunk)  # compile
    t0 = time.time()
    fut = generate(sampler, ctx, args.n_paths, key, args.chunk) * c["scale"]
    gen_s = time.time() - t0
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "samples.npy"), fut.astype(np.float32))
    label = args.label or ("ar_diffusion" if cfg["head"] == "diffusion" else f"ar_token{cfg['n_bins']}{cfg['bin_scheme'][0]}_{args.token_decode}")
    json.dump(dict(label=label, n_params=int(c["n_params"]), train_seconds=c["train_seconds"], gen_seconds=gen_s),
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(f"wrote {args.out}/samples.npy {fut.shape} in {gen_s:.1f}s")


if __name__ == "__main__":
    main()
