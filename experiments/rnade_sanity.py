"""
Practice runs for the RNADE option pricer from ECE_6257_Final_Project.ipynb.

The RNADE and pricer logic are copied from the notebook (same math), but here we
train on datasets where the TRUE option price is known, so we can check whether
the generated paths price options correctly.

Datasets (all are "% move per step", shape (D, N) like the notebook):
  toy  : the notebook's data. One uniform integer move in [-10, 10] repeated for
         all D steps (every path moves the same % each step).
  iid  : independent uniform integer moves in [-10, 10] at every step.
  gbm  : geometric Brownian motion (Black-Scholes world), converted to % moves.
         The true price has a closed form.

Usage:
  python experiments/rnade_sanity.py --dataset all
  python experiments/rnade_sanity.py --dataset gbm --epochs 20000 --hidden 64 --K 5
"""
import os as _os
import sys as _sys

if __package__ in (None, ""):  # also allow `python path/to/file.py`, not only `python -m pkg.file`
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import math
import os
import time

import torch
from torch import nn


# ---------------------------------------------------------------- model (from notebook)
class RNADE(nn.Module):
    def __init__(self, D, H, K=3):
        super().__init__()
        self.D, self.H, self.K = D, H, K
        self.W = nn.Parameter(torch.randn(H, D) * 0.01)
        self.c = nn.Parameter(torch.zeros(H))
        self.W_mu = nn.Parameter(torch.randn(D, K, H) * 0.01)
        self.b_mu = nn.Parameter(torch.zeros(D, K))
        self.W_si = nn.Parameter(torch.randn(D, K, H) * 0.01)
        self.b_si = nn.Parameter(torch.zeros(D, K))
        self.W_pi = nn.Parameter(torch.randn(D, K, H) * 0.01)
        self.b_pi = nn.Parameter(torch.zeros(D, K))
        self.min_si = 0.05

    def _params(self, X, i):
        h = torch.sigmoid(self.W[:, :i] @ X[:i, :] + self.c[:, None])
        mu = self.W_mu[i] @ h + self.b_mu[i, :, None]
        si = nn.functional.softplus(self.W_si[i] @ h + self.b_si[i, :, None]) + self.min_si
        logits = self.W_pi[i] @ h + self.b_pi[i, :, None]
        return mu, si, logits

    def log_density(self, X):
        result = 0.0
        for i in range(self.D):
            mu, si, logits = self._params(X, i)
            log_pi = torch.log_softmax(logits, dim=0)
            log_d = torch.distributions.Normal(mu, si).log_prob(X[i, :])
            result = result + torch.logsumexp(log_pi + log_d, dim=0)
        return result

    def fit(self, X, epochs=10000, lr=3e-4, log_every=1000):
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        hist = []
        for e in range(epochs):
            loss = -self.log_density(X).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if e % log_every == 0 or e == epochs - 1:
                print(f"  epoch {e:6d}  nll/path = {loss.item():.4f}", flush=True)
                hist.append((e, loss.item()))
        return hist

    @torch.no_grad()
    def generate(self, n):
        X = torch.zeros(self.D, n, device=self.W.device, dtype=self.W.dtype)
        cols = torch.arange(n, device=self.W.device)
        for i in range(self.D):
            mu, si, logits = self._params(X, i)
            k = torch.distributions.Categorical(logits=logits.T).sample()
            X[i, :] = mu[k, cols] + si[k, cols] * torch.randn(n, device=X.device)
        return X


# ---------------------------------------------------------------- pricing (from notebook wrapper)
def terminal_prices(moves_pct, S0):
    """moves_pct: (D, N) % moves -> terminal price per path, same formula as AutoOptPricer."""
    return S0 * torch.prod((moves_pct + 100) / 100, dim=0)


def price_payoffs(ST, K):
    call = (ST - K).clamp(min=0)
    put = (K - ST).clamp(min=0)
    straddle = call + put
    out = {}
    for name, v in [("call", call), ("put", put), ("straddle", straddle)]:
        out[name] = v.mean().item()
        out[name + "_se"] = (v.std() / math.sqrt(v.numel())).item()
    return out


# ---------------------------------------------------------------- datasets
def make_data(name, D, N, args, gen):
    if name == "toy":
        return torch.randint(-10, 11, (1, N), generator=gen).repeat(D, 1).float()
    if name == "iid":
        return torch.randint(-10, 11, (D, N), generator=gen).float()
    if name == "gbm":
        dt = 1.0 / 252
        z = torch.randn(D, N, generator=gen)
        logret = (args.mu - 0.5 * args.sigma ** 2) * dt + args.sigma * math.sqrt(dt) * z
        return 100 * (torch.exp(logret) - 1)
    raise ValueError(name)


def bs_undiscounted(S0, K, T, sigma, mu):
    """E[(S_T-K)^+] under GBM with drift mu (no discounting; matches the notebook pricer)."""
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    d1 = (math.log(S0 / K) + (mu + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S0 * math.exp(mu * T) * N(d1) - K * N(d2)
    put = call - (S0 * math.exp(mu * T) - K)  # parity on undiscounted expectations
    return {"call": call, "put": put, "straddle": call + put}


def true_prices(name, D, S0, K, args, gen):
    if name == "gbm":
        return bs_undiscounted(S0, K, D / 252, args.sigma, args.mu), "black-scholes closed form"
    big = make_data(name, D, args.truth_paths, args, gen)
    p = price_payoffs(terminal_prices(big, S0), K)
    return {k: p[k] for k in ("call", "put", "straddle")}, f"MC on true generator ({args.truth_paths:,} paths)"


def lag1_autocorr(X):
    a, b = X[:-1].flatten(), X[1:].flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# ---------------------------------------------------------------- main
def run(name, args, device):
    print(f"\n=== dataset: {name} ===", flush=True)
    gen = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    X = make_data(name, args.D, args.n_train, args, gen).to(device)

    model = RNADE(D=args.D, H=args.hidden, K=args.K).to(device)
    t0 = time.time()
    hist = model.fit(X, epochs=args.epochs, lr=args.lr)
    train_s = time.time() - t0

    G = model.generate(args.gen_paths)
    S0, K = args.S0, args.S0 * args.moneyness
    est = price_payoffs(terminal_prices(G, S0), K)
    nb_est = price_payoffs(terminal_prices(model.generate(500), S0), K)  # notebook's 500-path setting
    truth, truth_src = true_prices(name, args.D, S0, K, args, gen)
    emp = price_payoffs(terminal_prices(X, S0), K)  # price from the training set itself

    stats = {
        "train_mean_move": X.mean().item(), "gen_mean_move": G.mean().item(),
        "train_std_move": X.std().item(), "gen_std_move": G.std().item(),
        "train_lag1_autocorr": lag1_autocorr(X), "gen_lag1_autocorr": lag1_autocorr(G),
    }

    print(f"  trained in {train_s:.1f}s; truth = {truth_src}")
    print(f"  {'':10s}{'truth':>10s}{'train-set':>11s}{'RNADE':>10s}{'+-se':>8s}{'RNADE@500':>11s}{'rel err':>9s}")
    for p in ("call", "put", "straddle"):
        rel = (est[p] - truth[p]) / truth[p] if truth[p] else float("nan")
        print(f"  {p:10s}{truth[p]:10.4f}{emp[p]:11.4f}{est[p]:10.4f}{est[p+'_se']:8.4f}{nb_est[p]:11.4f}{rel:9.1%}")
    for k, v in stats.items():
        print(f"  {k:22s} {v: .4f}")

    return {"dataset": name, "truth_source": truth_src, "truth": truth, "train_set_price": emp,
            "rnade": est, "rnade_500paths": nb_est, "moment_stats": stats,
            "train_seconds": train_s, "loss_history": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all", "toy", "iid", "gbm"])
    ap.add_argument("--D", type=int, default=15, help="steps per path")
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--K", type=int, default=5, help="mixture components")
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gen_paths", type=int, default=200_000)
    ap.add_argument("--truth_paths", type=int, default=2_000_000)
    ap.add_argument("--S0", type=float, default=100.0)
    ap.add_argument("--moneyness", type=float, default=1.0, help="strike = S0 * moneyness")
    ap.add_argument("--sigma", type=float, default=0.20, help="gbm annual vol")
    ap.add_argument("--mu", type=float, default=0.0, help="gbm annual drift")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  torch {torch.__version__}")
    names = ["toy", "iid", "gbm"] if args.dataset == "all" else [args.dataset]
    results = [run(n, args, device) for n in names]

    os.makedirs(args.out, exist_ok=True)
    tag = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.out, f"rnade_sanity_{args.dataset}_{tag}.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), "device": device, "results": results}, f, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
