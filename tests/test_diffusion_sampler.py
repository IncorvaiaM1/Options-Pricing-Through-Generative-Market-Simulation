"""Exactness check for the DDPM sampler used by A1 (ardiff) and A6.

For x0 ~ N(0, 1) the ideal noise predictor is known in closed form, E[eps | x_t] = sqrt(1 - abar_t) x_t.
Plugging it into the sampler must give back a unit-variance Gaussian. This is what showed that the
"small" (beta tilde) variance shrinks the spread, which would bias option prices.

  python -m tests.test_diffusion_sampler
"""
import os as _os
import sys as _sys

if __package__ in (None, ""):  # also allow `python path/to/file.py`, not only `python -m pkg.file`
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from ardiff.diffusion import ddpm_sample, make_schedule


def run(T, var_type, n=200000):
    sched = make_schedule(T, "linear")
    ideal = lambda x, t, z: jnp.sqrt(1 - sched["abar"][t]) * x
    x = ddpm_sample(ideal, jnp.zeros((n, 1)), jax.random.PRNGKey(0), sched, var_type)
    return float(jnp.std(x)), float(jnp.mean(x))


def main():
    ok = True
    for T in (50, 100, 200):
        for v in ("large", "small"):
            sd, mu = run(T, v)
            print(f"T={T:4d} var={v:5s}  std={sd:.4f}  mean={mu:+.4f}")
            if v == "large":
                ok &= abs(sd - 1) < 0.01 and abs(mu) < 0.01
    print("PASS" if ok else "FAIL: 'large' variance sampler is not exact on Gaussian data")
    assert ok


if __name__ == "__main__":
    main()
