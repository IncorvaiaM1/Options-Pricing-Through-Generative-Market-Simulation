# ardiff: AR transformer + diffusion head (JAX)

**Thesis being tested:** Discrete-token AR models misprice deep OTM options and straddles because
tokenization rounds extreme returns too coarsely. An AR model with a continuous diffusion head
(MAR-style, Li et al. 2024) fixes this at similar compute.

## What's here

| file | what it does |
|---|---|
| `data.py` | GARCH(1,1) with Student-t innovations: fat tails + vol clustering, and an **exact truth**: given a context window, the GARCH state is known, so true option prices come from Monte Carlo from that state |
| `model.py` | Causal transformer backbone (continuous input embedding) + either a `DiffusionHead` (AdaLN-MLP denoiser, predicts noise on the next scalar return) or a `TokenHead` (softmax over bins) |
| `diffusion.py` | 1-D DDPM: schedule, training loss (several noise draws per token), ancestral sampler |
| `tokens.py` | Bins for the baseline: `quantile` (equal-mass) or `uniform` with clipping; decode to the training-data mean of each bin (default; fair to the wide edge bins), the bin center, or uniform jitter (`--token_decode`) |
| `train.py` | Training loop (optax AdamW, warmup + cosine) → `runs/<name>/ckpt.pkl` |
| `evaluate.py` | Generates paths from each checkpoint's eval contexts, prices OTM puts/calls/straddles across strikes, compares with truth in price and **implied-vol error**, splits out stressed contexts, logs cost (params, train time, µs per generated path) |

Both heads use the **same backbone and same continuous inputs**, so differences come only from the
output parameterization. (Kronos also tokenizes its inputs; that's a possible extra ablation.)

Strikes: `K = S0 * exp(k * sd_H)` with `sd_H` the long-run vol over the horizon. `k = -4..4`;
`k < 0` = OTM put, `k > 0` = OTM call, `k = 0` = straddle.

## Run

```bash
# local / interactive smoke test (CPU ok, a few minutes)
python -m ardiff.train --head diffusion --out runs/smoke_diff --n_train 2000 --steps 300 --d 64 --n_layers 2
python -m ardiff.train --head token     --out runs/smoke_tok  --n_train 2000 --steps 300 --d 64 --n_layers 2
python -m ardiff.evaluate --ckpts runs/smoke_diff/ckpt.pkl runs/smoke_tok/ckpt.pkl --n_paths 256 --n_mc_true 20000

# full first comparison on PACE (3 GPU training jobs + 1 eval job)
bash scripts/run_thesis_v0.sh
```

Note: all checkpoints in one `evaluate` call must share the data settings (`ctx_len, horizon, n_train, nu, data_seed`, ...).

## Defaults worth knowing

- context 64 days, horizon 20 days, GARCH alpha 0.08, beta 0.90, nu 5, 20% long-run vol
- backbone d=128, 4 layers, 4 heads; diffusion head width 256, depth 3, T=100 steps, linear schedule
- token baselines: 256 quantile bins, and 1024 uniform bins clipped at ±8 std

## Not done yet / next

- [ ] **Pure diffusion baseline** (no time structure): diffuse the whole 20-day future at once, conditioned on a context summary
- [ ] Kronos itself as a baseline (pretrained, real tokenizer) once we move to real data
- [ ] Ablations: n_bins (64 → 4096), token decode mode, diffusion steps at sampling (cost vs accuracy curve), nu (tail heaviness)
- [ ] Real data (SPX daily returns for training; option prices for evaluation)
- [ ] Risk-neutral question: right now everything is priced under the data-generating measure (r = 0, no discounting), which is fine for testing the thesis on synthetic data

## Smoke-test notes (CPU, tiny models, 2000 steps; not results)

- Pipeline runs end to end. Tiny models are far from converged, so the numbers mean nothing yet.
- Token head with quantile bins + center decoding hugely overpriced the tails: the edge bin spanned ~6 to 43 std and its center
  is ~24 std. Switched default decoding to the bin mean. Whatever we pick, the paper should report the decoding choice.
- Diffusion head: the cosine schedule's last reverse step amplifies by ~31x and blew samples up; switched to a scaled linear
  schedule. Still saw a rare 140-std sample with the undertrained head, so keep an eye on `step max abs` in the report.
