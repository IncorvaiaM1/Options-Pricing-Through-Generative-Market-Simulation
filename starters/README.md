# Starters: Track A (A2, A5, A6) and all of Track B

Every starter runs end to end, writes predictions in the shared format, and is scored by the shared harness,
so results from different people line up in one table. All of it is JAX/Flax except the numpy/scipy baselines
and the Schrödinger bridge. These are **starting points**: small, readable, defaults sized for a GPU node.

## 0. One-time data (both tracks)

```bash
python -m bench.trackA export --out bench_data/A            # GARCH-t paths + exact truth for Track A
python -m bench.surfaces --out bench_data/D2                # synthetic option surfaces + exact truth for Track B
python -m tests.test_diffusion_sampler                      # sanity check of the DDPM sampler (should print PASS)
bash scripts/smoke_all.sh                                   # toy-scale run of every starter (plumbing check only)
```

`bench_data/A` matches `ardiff/train.py` defaults, so A1/A2 checkpoints line up with it.
D2: 70% GJR-GARCH-t surfaces (truth by Monte Carlo with the empirical martingale correction, so the forward is
exactly 1) and 30% Heston surfaces (truth from the characteristic function). Strikes in std units k = -3..3,
maturities 5/10/20/40/60 days. Test quotes come in four fixed scenarios shared by every model:
`per_mat5` (5 strikes per maturity), `sparse10` (10 points total), `nowings` (|k| ≤ 1.5 only), `full` (all, for B2).

## Track A: market simulators → `samples.npy` → `bench.trackA score`

Any model, in any framework, writes `samples.npy` with shape `[E, N, H]` (future daily log returns for the E rows
of `bench_data/A/eval_ctx.npy`) plus an optional `meta.json`. Then:

```bash
python -m bench.trackA score --data bench_data/A --samples runs/<model>/samples.npy
python -m bench.trackA table "results/trackA/*.json"
```

Metrics: IV error at the money / in the wings / wings in stressed contexts, Asian and barrier price error,
vol clustering (lag-1 autocorrelation of |r|), explosive-sample rate, timing.

| id | file | what it is | run (GPU) |
|---|---|---|---|
| A1 | `ardiff/` | AR transformer + diffusion head (ours) | `bash scripts/run_thesis_v0.sh`, then `python -m ardiff.export_samples ...` |
| A2 | `ardiff/` + `scripts/a2_token_sweep.sh` | token AR, sweep of 64 to 4096 bins × quantile/uniform × 3 decode modes | `bash scripts/a2_token_sweep.sh`, later `--score` |
| A5 | `starters/a5_sde_gan.py` | neural SDE generator (Euler-Maruyama, latent state), GRU critic, WGAN-GP; `--jumps` ablation | `sbatch -J a5 scripts/pace_starter.sbatch starters.a5_sde_gan` |
| A6 | `starters/a6_pure_diffusion.py` | one DDPM over the whole 20-day future, context-conditioned, no AR | `sbatch -J a6 scripts/pace_starter.sbatch starters.a6_pure_diffusion` |

For teammates (MADE/RNADE, GAN): load `bench_data/A/train.npy` ([n, 84] = 64 context + 20 future days), train,
generate 8192 paths per row of `eval_ctx.npy`, `np.save("runs/<name>/samples.npy", arr)` in raw daily log-return
units, and score. For MADE, put the 64 context days first in the ordering and sample only the last 20.

## Track B: option-implied models → `preds/*.npz` → `bench.trackB score`

Each starter writes `results/trackB/preds/<method>_<scenario>.npz` and scores it immediately
(`results/trackB/<method>_<scenario>.json`). Leaderboard:

```bash
python -m bench.trackB table "results/trackB/*.json"
```

Metrics: held-out IV RMSE (all, wings, per family), 90% band coverage, Wasserstein-1 to the true density,
butterfly-arbitrage rate, exotic price error and band coverage.

| id | file | what it is | run |
|---|---|---|---|
| baselines | `starters/b_baselines.py` | linear interp, SSVI, GP (with bands), Heston calibration, relaxed max-entropy (Buchen & Kelly) | CPU: `python -m starters.b_baselines` |
| B1 | `starters/b1_ebm.py` | amortized EBM: DeepSets quote encoder, energy on a grid, exact normalizer, martingale tilt | GPU |
| B1 | `starters/b1_flow.py` | per-maturity monotone 1-D flow fitted to quotes, exact density, exact forward | GPU or CPU |
| B2 | `starters/b2_neural_sde.py` | neural local-stochastic-vol SDE calibrated to the full surface; ensemble band on 6 exotics; vs calibrated Heston | GPU |
| B3 | `starters/b3_neural_process.py` | attentive conditional NP: surface completion with a Gaussian band | GPU |
| B3 | `starters/b3_vae.py` | surface VAE: latent factor R², completion by latent fit, latent stress walks | GPU (small) |
| B4 | `starters/b4_schrodinger_bridge.py` | martingale Schrödinger bridge 20d→60d (Sinkhorn + Newton), forward-start options vs truth | CPU |

B4 uses the true D2 densities by default; `--density_pred results/trackB/preds/ebm_per_mat5.npz` runs it on a
B1 model's densities instead (the end-to-end version).

## What the toy-scale smoke runs showed (plumbing checks, not results)

- **DDPM variance matters for pricing.** With a perfect denoiser, the common "small" reverse variance shrinks the
  sample std by 4% at T=100 (8% at T=50), a direct vol bias. Both A1 and A6 now default to the "large" variance,
  which is exact on Gaussian data (`tests/test_diffusion_sampler.py`).
- **A6** needs the full 20k training paths (it overfit and blew up on 4k). It trains on asinh-transformed returns
  and clips the predicted clean sample at 1.5x the training max (in asinh space, i.e. hundreds of std). At 20k
  paths: ~1.2 vol pts ATM error, ~2.7 in the wings, and lag-1 |r| autocorrelation 0.02 vs 0.12 true, which is
  the H5 prediction (no AR structure, no vol clustering).
- **Max-entropy** must use the relaxed form with noisy quotes; exact matching made densities worse with more quotes.
- **B4**: the ATM forward-start straddle is too easy (plain Black gets ~1%); the forward-start 90% put is the
  informative test (bridge ~3% vs Black ~9% on 8 surfaces).
- **B2** at toy scale: calibration not converged and the 4-member ensemble band is too narrow to cover the truth.
  Expect to need more steps, more members, or a robust-bounds objective as in Gierjatowicz et al.

## Known gaps / next steps

- A5 critic is a GRU; Kidger et al. use a neural CDE (diffrax). Worth swapping once the GRU version trains.
- B3 NP is the deterministic (CNP) path; add a latent path if bands are over-confident on unusual surfaces.
- B4 needs Dupire local vol as the classic comparator, and a multi-maturity version.
- Calendar-arbitrage check is not in the Track B scorer yet (butterfly only).
- The D2 GARCH family is the "Heston is wrong" world for H7; the Heston family is the control.
