# ECE 6257 Final Project (working title TBD)

Final project for ECE 6257: Generative and Geometric Deep Learning, Georgia Tech, Fall 2026.
Deliverable: a 4-6 page ICML-style short paper.

**Status:** brainstorming. Nothing is decided yet.

## Team

- Michael Incorvaia
- TBD
- TBD

## Theme

Generative models for finance, likely option pricing or market scenario generation.

## Candidate directions (not committed)

1. **AR + diffusion hybrid pricer**: autoregressive transformer over time with a continuous diffusion head per step; test whether it prices tails/straddles better than tokenized AR models (e.g. Kronos).
2. **Risk-neutral fine-tuning of an AR market model**: show pretrained market simulators misprice options, then calibrate them to observed option prices.
3. **Symmetry-aware / arbitrage-free surface generation**: option chains as point sets, no-arbitrage enforced by construction.
4. **Generative metrics vs financial metrics**: do standard sample-quality metrics rank generators the same way pricing/hedging error does?
5. **Conditional pricing with exogenous data**: weather, futures, VIX as conditioning (data permitting).

## Open questions

- Which direction / thesis?
- Data source (WRDS/OptionMetrics access? synthetic Heston/SABR first?)
- Compute (Colab, CoC-ICE)

## Repo layout

TBD once a direction is chosen.
