# Running on PACE ICE

## 1. SSH in

Off campus, connect to the GT VPN (GlobalProtect) first. Then from PowerShell:

```powershell
ssh <gt-username>@login-ice.pace.gatech.edu
```

Use your GT password, then approve the Duo push.

## 2. One-time setup

```bash
git clone https://github.com/IncorvaiaM1/Options-Pricing-Through-Generative-Market-Simulation.git
cd Options-Pricing-Through-Generative-Market-Simulation

module load anaconda3
conda create -n gdl python=3.11 -y
conda activate gdl
pip install -r requirements.txt
pip install -U "jax[cuda12]" flax optax   # for the ardiff (JAX) code
python -c "import jax; print(jax.devices())"   # on a GPU node this should list a CudaDevice
```

Home has a small quota; if installs fail for space, put the env in scratch:
`conda create -p ~/scratch/envs/gdl python=3.11 -y` and `conda activate ~/scratch/envs/gdl`.

## 3. Run

Quick interactive test (don't run heavy jobs on the login node):

```bash
salloc -N1 --ntasks-per-node=4 -t 0:30:00        # add --gres=gpu:1 for a GPU
module load anaconda3 && conda activate gdl
python experiments/rnade_sanity.py --dataset gbm --epochs 2000
exit
```

Batch job:

```bash
sbatch scripts/pace_rnade_sanity.sbatch                     # all three datasets
sbatch scripts/pace_rnade_sanity.sbatch --dataset gbm --hidden 64 --epochs 20000
squeue -u $USER                                            # check status
tail -f logs/rnade_sanity-<jobid>.out                      # watch output
```

Results land in `results/*.json`.

## JAX AR + diffusion experiments

```bash
bash scripts/run_thesis_v0.sh         # 3 training jobs + eval job that waits for them
cat results/v0/report.md              # when done
```

See `ardiff/README.md` for what each piece does.

## Model comparison workflow (pitch-aligned)

```bash
git pull && pip install -r requirements.txt          # adds yfinance + pandas

# synthetic benchmark (exact truth)
python -m bench.trackA export --out bench_data/A
bash scripts/submit_all.sh bench_data/A              # one GPU job per model + a report job

# real data: run the setup on the LOGIN node (compute nodes may not have internet)
bash scripts/real_data_setup.sh megacap_tech         # yfinance download, time-split export, today's option chains
MARKET=bench_data/market/$(date +%F)_megacap_tech bash scripts/submit_all.sh bench_data/real/megacap_tech

squeue -u $USER                                      # progress; figures land in figs/<data>/ when the report job runs
```

One model by hand: `sbatch -J a6 scripts/pace_model.sbatch pure_diffusion bench_data/A`.
Plumbing check before a big run: `QUICK=1 bash scripts/run_model.sh <model> bench_data/A` on an interactive node.

## Hedging metric (optional, needs fatnarwhal-quant)

Until the package has a public repo, copy it to PACE (e.g. `scp -r fatnarwhal-quant <gt-user>@login-ice.pace.gatech.edu:~/`)
and `pip install -e ~/fatnarwhal-quant`. Re-export the synthetic data so it keeps enough true paths for hedging:
`python -m bench.trackA export --out bench_data/A --n_hedge_paths 10000`. The scorers pick the metric up automatically.

## Notebook (live view of all models)

```bash
conda activate gdl
pip install ipykernel && python -m ipykernel install --user --name gdl --display-name "Python (gdl)"
```
Then OnDemand (ondemand-ice.pace.gatech.edu) -> Interactive Apps -> Jupyter, request a GPU, open
`notebooks/run_all.ipynb`, pick the "Python (gdl)" kernel, run top to bottom. On a login-node session set `MODE = "slurm"`
instead (it submits jobs and watches them; training never runs on the login node).
Clear outputs before committing the notebook (Edit -> Clear All Outputs) so the repo stays small.
