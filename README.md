# Options Pricing Through Generative Market Simulation

ECE 6257 (Generative and Geometric Deep Learning) final project, Georgia Tech, Fall 2026.
Kian Chakamian, Michael Incorvaia, Rajan Sandhu, Yosep Rhe. Deadline: December 7.

**Idea (from the pitch):** train generative models on historical price movements, generate many future price paths,
and price options as the discounted expected payoff. Compare model families against each other, against
Black-Scholes-Merton, against what actually happened, and against current market prices.

**Thesis we test inside that comparison:** discrete-token autoregressive models misprice deep out-of-the-money
options because tokenization rounds extreme returns too coarsely; an AR model with a continuous diffusion head fixes
this at similar compute.

## Models (Track A: all trained on price history, all scored the same way)

| name in scripts | what | code |
|---|---|---|
| `bsm`, `bootstrap`, `garch` | classical baselines: Black-Scholes-Merton at realized vol, historical bootstrap, GARCH-t (MLE) | `starters/a0_baselines.py` |
| `ar_diffusion` | **A1**, ours: causal transformer + per-step diffusion head (JAX) | `ardiff/` |
| `ar_token` | **A2**, Kronos-style tokenized AR, same backbone | `ardiff/` |
| `rnade` | **A3**, the team's RNADE (PyTorch), conditioned by putting the 64 context days first | `starters/a3_rnade.py` |
| `gan` | **A4**, conditional WGAN-GP (placeholder until the teammate GAN is plugged in) | `starters/a4_gan.py` |
| `sde_gan` | **A5**, neural SDE generator trained as a GAN | `starters/a5_sde_gan.py` |
| `pure_diffusion` | **A6**, one DDPM over the whole 20-day path (no AR) | `starters/a6_pure_diffusion.py` |
| `path_flow`, `path_flow_t` | **A7**, conditional normalizing flow over the path (Gaussian / Student-t base) | `starters/a7_path_flow.py` |

Track B (option-implied models, outside the pitch, kept for now): see `starters/README.md`.

## Data

- **Synthetic (exact truth):** `python -m bench.trackA export --out bench_data/A`. GARCH-t paths; true option prices
  known for every context, so pricing error is measured exactly.
- **Real (yfinance):** `bash scripts/real_data_setup.sh megacap_tech` on a machine with internet (PACE login node).
  Downloads adjusted closes + T-bill rates, makes a time-split export (train to 2021, validate to mid 2023, test after),
  and takes a snapshot of today's option chains. Presets: `spy`, `index_etfs`, `megacap_tech`, `banks`, `energy`,
  `staples` (clusters of similar stocks), or `TICKERS="AAPL"` for a single name.

Every script runs either way from the repo root: `python -m starters.b4_schrodinger_bridge` or
`python starters/b4_schrodinger_bridge.py`. Track B scripts need `python -m bench.surfaces --out bench_data/D2` first.

## Run one model

```bash
bash scripts/run_model.sh <model> <data_dir>                        # e.g. pure_diffusion bench_data/A
MARKET=bench_data/market/2026-10-27_megacap_tech bash scripts/run_model.sh path_flow bench_data/real/megacap_tech
QUICK=1 bash scripts/run_model.sh rnade bench_data/A                  # tiny settings, plumbing check
```

Each run trains, generates 8192 paths per evaluation context, scores, and plots:
- synthetic: `results/A/trackA/<label>.json` (IV error by strike, wings, stressed markets, path stats)
- real: `results/<name>/backtest/<label>.json` (vs realized payoffs, calibration, CRPS, discounted EV prices)
- market: `results/<name>/market/<label>.json` (implied vol vs market mid, risk-neutral adjusted)
- `figs/<data>/paths_<label>.png` (simulated price paths, like the notebook plot, with bands and truth / reality)

## Run everything from a notebook

`notebooks/run_all.ipynb` runs any model, set of models (`baselines`, `thesis`, `generative`, `all`) or Track B method
from cells, using the same scripts as above. While a model trains you see its log tail and its loss curves grow; after
each one the leaderboard table, IV-error-by-strike comparison and its price-path plot update. Full logs go to
`logs/notebook/`. Open it in a PACE OnDemand Jupyter session on a GPU node with the `gdl` kernel (`MODE = "local"`), or
set `MODE = "slurm"` to submit batch jobs and watch all of them live from one cell. `QUICK = True` checks the plumbing in
about 15 minutes on CPU. Helpers live in `bench/nb.py`.

## Run everything on PACE

```bash
bash scripts/submit_all.sh bench_data/A                                            # every model, one GPU job each
MARKET=bench_data/market/2026-10-27_megacap_tech bash scripts/submit_all.sh bench_data/real/megacap_tech   # use the folder real_data_setup.sh printed
MODELS="pure_diffusion path_flow" bash scripts/submit_all.sh bench_data/A          # a subset
bash scripts/report.sh bench_data/A                                                # tables + figures (auto after submit_all)
```

`scripts/report.sh` writes leaderboard tables and figures to `figs/<data>/`: IV error by strike, smiles vs truth,
terminal-return densities (tails), backtest summary and PIT histograms, market smiles. Setup details: `PACE.md`.

## Hedging metric (fatnarwhal-quant)

The Black-Scholes, Greeks and implied vol behind FatNarwhal's site tools ship as a small library,
`fatnarwhal-quant`. With it installed, every score also reports **hedging error**: sell the option at the model's
price, delta-hedge daily at the model's implied vol on the TRUE paths, pay the payoff. Mean P&L is the pricing error a
trader would feel; the std is hedge quality. Both are reported against the oracle (true price and true vol hedged on
the same paths), for every Track A model (`bench/trackA.py`) and for Track B surfaces at 20 days (`bench/trackB.py`).
Without the package the scorers skip this metric and say so.

```bash
pip install -e <path to fatnarwhal-quant>      # until the public repo exists; then the line in requirements.txt
python -m pytest tests/test_fn_quant_crosscheck.py   # bench/finance.py agrees with the site's math (18 tests)
python -m experiments.hedging_demo                   # oracle vs Black-Scholes hedging, ~30 s on CPU
```

Notes for the paper: the oracle is a reference, not a floor (a Black-Scholes delta at the true implied vol is not
the variance-minimizing hedge under fat tails, so ratios slightly below 1 happen); use `--n_hedge_paths 10000` in
`bench.trackA export` for final numbers, since deep-wing strikes are rarely exercised.

**Known issue (team decision pending):** the synthetic GARCH-t spot is not a martingale. Log returns are mean zero, so
E[S_T] drifts up by about half the summed variance (~15 to 18 bps over 20 days). Model-vs-truth comparisons are still
fair (same measure on both sides); hedging is reported against the oracle for this reason. A true fix (convexity-corrected,
truncated innovations) changes the truth and means rerunning everything. The Track B surfaces (`bench/surfaces.py`)
and the real-data market comparison are already martingale-corrected.

## Teammate models

Any model in any framework plugs in by writing `samples.npy`, shape `[E, N, 20]` = future daily log returns for the
E rows of `<data_dir>/eval_ctx.npy` (training windows are in `<data_dir>/train.npy`, `[n, 84]` = 64 context + 20
future days). Then score with `python -m bench.trackA score` (synthetic) or `python -m bench.backtest` (real).

## Layout

```
bench/      data, scoring and plots: trackA (synthetic), real + backtest + market (yfinance), surfaces + trackB, plots
ardiff/     A1/A2 (JAX)          starters/   A0, A3-A7, Track B models          tests/   sampler exactness test
scripts/    run_model.sh, submit_all.sh, report.sh, real_data_setup.sh, PACE sbatch files, smoke_all.sh
docs/       pitch, plan of action
```
