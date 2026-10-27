"""A3: RNADE (real-valued NADE with Gaussian-mixture conditionals), PyTorch, same math as the team notebook.

Conditioning by ordering: each training window is the 64 context days followed by the 20 future days, treated as
one (L+H)-dimensional vector in time order. RNADE models it autoregressively, so to generate a future for a
given context we clamp the first L dimensions to the context and sample only the last H. No architecture change
from the notebook; the only additions are the incremental hidden-state update (a_i = a_{i-1} + W[:, i-1] x_{i-1},
the standard NADE trick, O(D) instead of O(D^2)), minibatching, and a GPU option.

  python -m starters.a3_rnade --data bench_data/A --out runs/a3
  python -m bench.trackA score --data bench_data/A --samples runs/a3/samples.npy
"""
import os as _os
import sys as _sys

if __package__ in (None, ""):  # also allow `python path/to/file.py`, not only `python -m pkg.file`
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn


class RNADE(nn.Module):
    def __init__(self, D, H, K=5):
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

    def _cond(self, a, i):
        h = torch.sigmoid(a)                                       # [B, H]
        mu = h @ self.W_mu[i].T + self.b_mu[i]                     # [B, K]
        si = nn.functional.softplus(h @ self.W_si[i].T + self.b_si[i]) + self.min_si
        logits = h @ self.W_pi[i].T + self.b_pi[i]
        return mu, si, logits

    def log_density(self, X):
        """X: [B, D] -> log p(x) [B]."""
        a = self.c.expand(X.shape[0], -1)
        ll = 0.0
        for i in range(self.D):
            mu, si, logits = self._cond(a, i)
            lp = torch.log_softmax(logits, -1) + torch.distributions.Normal(mu, si).log_prob(X[:, i:i + 1])
            ll = ll + torch.logsumexp(lp, -1)
            a = a + X[:, i:i + 1] * self.W[:, i]
        return ll

    @torch.no_grad()
    def sample_given_prefix(self, prefix, n_total):
        """prefix: [B, L] clamped dims; returns samples of the remaining D-L dims, [B, D-L]."""
        B, L = prefix.shape
        a = self.c.expand(B, -1).clone()
        for i in range(L):
            a = a + prefix[:, i:i + 1] * self.W[:, i]
        out = torch.empty(B, self.D - L, device=prefix.device)
        for i in range(L, self.D):
            mu, si, logits = self._cond(a, i)
            k = torch.distributions.Categorical(logits=logits).sample()
            x = mu.gather(1, k[:, None])[:, 0] + si.gather(1, k[:, None])[:, 0] * torch.randn(B, device=prefix.device)
            out[:, i - L] = x
            a = a + x[:, None] * self.W[:, i]
        return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench_data/A")
    ap.add_argument("--out", default="runs/a3")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_paths", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--also_eval", nargs="*", default=[], help="extra data folders to generate samples for (e.g. a market snapshot)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(args.data, "config.json")))
    s = cfg["scale"]
    L, H = cfg["ctx_len"], cfg["horizon"]
    train = torch.tensor(np.load(os.path.join(args.data, "train.npy")) / s, dtype=torch.float32, device=dev)
    val = torch.tensor(np.load(os.path.join(args.data, "val.npy"))[:2048] / s, dtype=torch.float32, device=dev)
    ctx = torch.tensor(np.load(os.path.join(args.data, "eval_ctx.npy")) / s, dtype=torch.float32, device=dev)
    model = RNADE(L + H, args.hidden, args.K).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * 0.05)
    print(f"A3 RNADE on {dev}: {sum(p.numel() for p in model.parameters()):,} params, D={L + H}", flush=True)
    t0 = time.time()
    for it in range(1, args.steps + 1):
        b = train[torch.randint(0, train.shape[0], (args.batch,), device=dev)]
        loss = -model.log_density(b).mean() / (L + H)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % args.log_every == 0 or it == 1:
            with torch.no_grad():
                vl = -model.log_density(val).mean() / (L + H)
            print(f"step {it:6d}  train NLL/dim {loss.item():.4f}  val NLL/dim {vl.item():.4f}  {time.time() - t0:.0f}s", flush=True)
    train_s = time.time() - t0
    t0 = time.time()
    rep = ctx.repeat_interleave(args.n_paths, 0)
    outs = [model.sample_given_prefix(rep[i:i + args.chunk], L + H).cpu().numpy() for i in range(0, rep.shape[0], args.chunk)]
    gen_s = time.time() - t0
    fut = np.concatenate(outs).reshape(ctx.shape[0], args.n_paths, H) * s
    meta = dict(label="a3_rnade", n_params=int(sum(p.numel() for p in model.parameters())), train_seconds=train_s, gen_seconds=gen_s)

    def save(out, f):
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "samples.npy"), f.astype(np.float32))
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
        print(f"wrote {out}/samples.npy {f.shape}")

    save(args.out, fut)
    for d in args.also_eval:
        c2 = torch.tensor(np.load(os.path.join(d, "eval_ctx.npy")) / s, dtype=torch.float32, device=dev)
        rep = c2.repeat_interleave(args.n_paths, 0)
        o = [model.sample_given_prefix(rep[i:i + args.chunk], L + H).cpu().numpy() for i in range(0, rep.shape[0], args.chunk)]
        save(os.path.join(args.out, "also_" + os.path.basename(os.path.normpath(d))),
             np.concatenate(o).reshape(c2.shape[0], args.n_paths, H) * s)


if __name__ == "__main__":
    main()
