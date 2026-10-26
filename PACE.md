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
