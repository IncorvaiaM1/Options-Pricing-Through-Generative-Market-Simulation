#!/bin/bash
# Runs every starter at toy scale end to end (CPU is fine, ~15-25 min on a laptop; a few min on a GPU node).
# Numbers from this are NOT results; it only proves the plumbing works. Outputs go to smoke_*/ (gitignored).
#   bash scripts/smoke_all.sh
set -e
cd "$(dirname "$0")/.."
D=smoke_data; R=smoke_runs; O=smoke_results
mkdir -p $D $R $O

echo "== data =="
python -m bench.trackA export --out $D/A --n_eval 8 --n_mc 20000
python -m bench.surfaces --out $D/D2 --n_train 64 --n_test 12 --paths 5000 --test_paths 30000
python -m tests.test_diffusion_sampler
python -m pytest -q tests/test_fn_quant_crosscheck.py || true   # skips if fatnarwhal-quant is not installed

echo "== Track A =="
python -m starters.a6_pure_diffusion --data $D/A --out $R/a6 --width 128 --depth 2 --T 200 --steps 5000 --log_every 1000 --n_paths 256
python -m starters.a5_sde_gan --data $D/A --out $R/a5 --dh 16 --dw 8 --width 64 --substeps 2 --steps 200 --batch 128 --log_every 100 --n_paths 256
python -m ardiff.train --head token --out $R/a2_tok --n_eval 8 --steps 300 --d 64 --n_layers 2 --batch 128 --log_every 150
python -m ardiff.export_samples --ckpt $R/a2_tok/ckpt.pkl --data $D/A --out $R/a2_tok --n_paths 256
for m in a6 a5 a2_tok; do python -m bench.trackA score --data $D/A --samples $R/$m/samples.npy --out $O/trackA; done
python -m bench.trackA table "$O/trackA/*.json"

echo "== Track B =="
B="--test $D/D2/test.npz --pred_dir $R/predsB --out $O/trackB"
python -m starters.b_baselines $B
python -m starters.b1_ebm --train $D/D2/train.npz $B --width 64 --steps 300 --batch 32 --log_every 150
python -m starters.b1_flow $B --steps 300
python -m starters.b3_neural_process --train $D/D2/train.npz $B --width 64 --steps 500 --batch 32 --log_every 250
python -m starters.b3_vae --train $D/D2/train.npz $B --steps 1000 --log_every 500 --fit_steps 100
python -m starters.b2_neural_sde $B --n_surfaces 2 --ensemble 2 --width 32 --paths 512 --price_paths 5000 --steps 50
python -m starters.b4_schrodinger_bridge --test $D/D2/test.npz --out $O/trackB --n_surfaces 4
python -m bench.trackB table "$O/trackB/*_per_mat5.json" "$O/trackB/*_sparse10.json" "$O/trackB/*_nowings.json"
echo "smoke test OK"
