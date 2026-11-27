# megacap_tech, run `full2`

Started 2026-11-26T17:24:03, data `bench_data/real/megacap_tech` (real). Collected 2026-11-26T19:20:24. Only files written after the start are included.

> **Warning:** fatnarwhal_quant not importable: hedging columns will be empty (pip install -e <fatnarwhal-quant>)

## Models

| model | status | min | label | params | train s | gen s | log | last error |
|---|---|---|---|---|---|---|---|---|
| bsm | ok | 0.7 | baseline_bsm | 0 | 0 | 2.8 | [log](logs/bsm.log) |  |
| bootstrap | ok | 0.2 | baseline_bootstrap | 0 | 0 | 2.3 | [log](logs/bootstrap.log) |  |
| garch | ok | 0.5 | baseline_garch | 4 | 0 | 10.6 | [log](logs/garch.log) |  |
| ar_diffusion | ok | 42.5 | ar_diffusion | 2,088,321 | 275 | 1,192.0 | [log](logs/ar_diffusion.log) |  |
| ar_token | ok | 22.1 | ar_token256q_mean | 837,504 | 97 | 644.4 | [log](logs/ar_token.log) |  |
| rnade | ok | 27.6 | a3_rnade | 87,340 | 1,500 | 4.3 | [log](logs/rnade.log) |  |
| gan | ok | 7.4 | a4_gan | 293,908 | 271 | 2.7 | [log](logs/gan.log) |  |
| sde_gan | ok | 8.7 | a5_sde_gan | 194,657 | 465 | 17.6 | [log](logs/sde_gan.log) |  |
| pure_diffusion | ok | 4.8 | a6_pure_diffusion | 3,916,052 | 60 | 130.5 | [log](logs/pure_diffusion.log) |  |
| path_flow_t | ok | 1.3 | a7_path_flow_t | 938,944 | 54 | 3.7 | [log](logs/path_flow_t.log) |  |

## Leaderboard

| model | crps_mean | coverage90 | exceed_1pct | exceed_5pct | qlike | worst |payoff bias t| |
|---|---|---|---|---|---|---|
| a3_rnade | 0.04608 | 0.8937 | 0.009202 | 0.04703 | -3.932 | 4.1 |
| baseline_garch | 0.04808 | 0.909 | 0.001022 | 0.03476 | -3.886 | 12.7 |
| baseline_bsm | 0.04897 | 0.8906 | 0.00818 | 0.03476 | -3.902 | 9.9 |
| ar_token256q_mean | 0.04911 | 0.8793 | 0.01022 | 0.07055 | -3.924 | 3.3 |
| a5_sde_gan | 0.05345 | 0.6084 | 0.1431 | 0.2239 | -2.286 | 7.0 |
| ar_diffusion | 0.05367 | 0.5082 | 0.1677 | 0.2638 | -0.2951 | 9.4 |
| baseline_bootstrap | 0.05419 | 0.8456 | 0.03885 | 0.091 | -3.899 | 5.9 |
| a4_gan | 0.05698 | 0.5859 | 0.136 | 0.2055 | -1.933 | 5.3 |
| a6_pure_diffusion | 0.05954 | 0.3016 | 0.2965 | 0.3548 | 12.73 | 10.0 |
| a7_path_flow_t | 0.0637 | 0.5706 | 0.1115 | 0.2239 | -1.759 | 5.4 |

## Comparison

**backtest_pit**

![backtest_pit.png](figs/backtest_pit.png)

**backtest_summary**

![backtest_summary.png](figs/backtest_summary.png)

**market_by_moneyness**

![market_by_moneyness.png](figs/market_by_moneyness.png)

**market_smiles**

![market_smiles.png](figs/market_smiles.png)

**terminal_distributions**

![terminal_distributions.png](figs/terminal_distributions.png)

## Training curves

**loss_all**

![loss_all.png](figs/loss_all.png)

**loss_ar_diffusion**

![loss_ar_diffusion.png](figs/loss_ar_diffusion.png)

**loss_ar_token**

![loss_ar_token.png](figs/loss_ar_token.png)

**loss_gan**

![loss_gan.png](figs/loss_gan.png)

**loss_path_flow_t**

![loss_path_flow_t.png](figs/loss_path_flow_t.png)

**loss_pure_diffusion**

![loss_pure_diffusion.png](figs/loss_pure_diffusion.png)

**loss_rnade**

![loss_rnade.png](figs/loss_rnade.png)

**loss_sde_gan**

![loss_sde_gan.png](figs/loss_sde_gan.png)

## Simulated paths

**paths_a3_rnade**

![paths_a3_rnade.png](figs/paths_a3_rnade.png)

**paths_a4_gan**

![paths_a4_gan.png](figs/paths_a4_gan.png)

**paths_a5_sde_gan**

![paths_a5_sde_gan.png](figs/paths_a5_sde_gan.png)

**paths_a6_pure_diffusion**

![paths_a6_pure_diffusion.png](figs/paths_a6_pure_diffusion.png)

**paths_a7_path_flow_t**

![paths_a7_path_flow_t.png](figs/paths_a7_path_flow_t.png)

**paths_ar_diffusion**

![paths_ar_diffusion.png](figs/paths_ar_diffusion.png)

**paths_ar_token256q_mean**

![paths_ar_token256q_mean.png](figs/paths_ar_token256q_mean.png)

**paths_baseline_bootstrap**

![paths_baseline_bootstrap.png](figs/paths_baseline_bootstrap.png)

**paths_baseline_bsm**

![paths_baseline_bsm.png](figs/paths_baseline_bsm.png)

**paths_baseline_garch**

![paths_baseline_garch.png](figs/paths_baseline_garch.png)

