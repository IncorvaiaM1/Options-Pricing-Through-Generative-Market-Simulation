"""1-D conditional DDPM utilities (epsilon prediction; linear or cosine schedule)."""
import jax
import jax.numpy as jnp
import numpy as np


def make_schedule(T, kind="linear"):
    """'linear' = DDPM linear betas rescaled to T steps (beta_max = 0.02 * 1000/T, capped at 0.5).
    Keeps every reverse step's 1/sqrt(alpha) amplification small, which matters for heavy-tailed
    targets fed back autoregressively. 'cosine' kept for ablation."""
    if kind == "cosine":
        return cosine_schedule(T)
    scale = 1000.0 / T
    betas = np.linspace(1e-4 * scale, min(0.02 * scale, 0.5), T)
    alphas = 1 - betas
    abar = np.cumprod(alphas)
    return dict(betas=jnp.asarray(betas, jnp.float32), alphas=jnp.asarray(alphas, jnp.float32),
                abar=jnp.asarray(abar, jnp.float32), T=T)


def cosine_schedule(T, s=0.008):
    steps = np.arange(T + 1, dtype=np.float64) / T
    ab = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    ab = ab / ab[0]
    betas = np.clip(1 - ab[1:] / ab[:-1], 1e-8, 0.999)
    alphas = 1 - betas
    abar = np.cumprod(alphas)
    return dict(betas=jnp.asarray(betas, jnp.float32), alphas=jnp.asarray(alphas, jnp.float32),
                abar=jnp.asarray(abar, jnp.float32), T=T)


def diffusion_loss(eps_fn, z, x0, key, sched, mul=4):
    """z: [N, d] conditions, x0: [N] targets. `mul` noise draws per target (MAR trick) to cut variance."""
    N = x0.shape[0]
    z = jnp.repeat(z, mul, axis=0)
    x0 = jnp.repeat(x0, mul, axis=0)
    k1, k2 = jax.random.split(key)
    t = jax.random.randint(k1, (N * mul,), 0, sched["T"])
    eps = jax.random.normal(k2, x0.shape)
    ab = sched["abar"][t]
    x_t = jnp.sqrt(ab) * x0 + jnp.sqrt(1 - ab) * eps
    pred = eps_fn(x_t, t, z)
    return jnp.mean((pred - eps) ** 2)


def ddpm_sample(eps_fn, z, key, sched, var_type="large", clip=None):
    """Ancestral DDPM sampling of one scalar per row of z. z: [N, d] -> x0: [N].

    var_type: "large" uses sigma_t^2 = beta_t, "small" uses the posterior variance beta_tilde_t.
    With a perfect denoiser on unit-Gaussian data, "small" shrinks the sample std to 0.96 at T=100
    (0.92 at T=50) while "large" stays at 1.00 (see tests/test_diffusion_sampler.py). For option pricing
    that shrinkage is a direct vol bias, so "large" is the default.
    clip: optional bound on the predicted clean sample x0_hat (None = no clipping)."""
    N = z.shape[0]
    T = sched["T"]
    key, k0 = jax.random.split(key)
    x = jax.random.normal(k0, (N,))

    def step(carry, t):
        x, key = carry
        key, kn = jax.random.split(key)
        tt = jnp.full((N,), t, jnp.int32)
        e = eps_fn(x, tt, z)
        a, b, ab = sched["alphas"][t], sched["betas"][t], sched["abar"][t]
        ab_prev = jnp.where(t > 0, sched["abar"][jnp.maximum(t - 1, 0)], 1.0)
        x0 = (x - jnp.sqrt(1 - ab) * e) / jnp.sqrt(ab)
        if clip is not None:
            x0 = jnp.clip(x0, -clip, clip)
        mean = (jnp.sqrt(ab_prev) * b * x0 + jnp.sqrt(a) * (1 - ab_prev) * x) / (1 - ab)
        var = b if var_type == "large" else b * (1 - ab_prev) / (1 - ab)
        x = mean + jnp.where(t > 0, jnp.sqrt(var), 0.0) * jax.random.normal(kn, (N,))
        return (x, key), None

    (x, _), _ = jax.lax.scan(step, (x, key), jnp.arange(T - 1, -1, -1))
    return x
