import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int

from pkg.scheduler import DiffusionScheduler
from pkg.transformer import SequenceDiT


@eqx.filter_jit
# @eqx.debug.assert_max_traces(max_traces=2)
def diffusion_lm_loss(
    key:              jax.random.PRNGKey,
    denoiser:         SequenceDiT,                           # f_θ
    tokens:           Int[Array, "batch seq"],
    max_t:            int,                                  # T
    cond_emb:         Float[Array, "batch cond_dim"],       # e.g. “dreams_embeddings”
    sched:            DiffusionScheduler,
) -> Array:
    """Implements  L_e2e^{x0-simple}  (Li et al. 2022, Eq. 19 / App.F)"""

    # ---- PRNG bookkeeping --------------------------------------------------
    rng_noise, rng_T, rng_t, rng_1 = jax.random.split(key, 4)

    batch_size = tokens.shape[0]

    # ---- sample x0  ~  q_φ(x0 | w)  ----------------------------------------
    sigma0 = sched.sqrt_alpha_cumprod_0                                 # small (learned) σ₀
    emb_clean, x0 = denoiser.embedding(tokens.reshape(-1), noise_std=sigma0, key=rng_noise)
    emb_clean = emb_clean.reshape(batch_size, -1, denoiser.embedding_size)
    x0 = x0.reshape(batch_size, -1, denoiser.embedding_size)

    # ---- KL-to-prior term at t = T -----------------------------------------
    xT     = sched.add_noise(rng_T, x0, max_t)            # x_T
    mu_hat = sched.posterior_mean(xT, x0, max_t)         # \hat{μ}_T
    kl_T   = jnp.mean(jnp.sum(mu_hat**2, axis=(-1, -2)))  # ||·||²

    # ---- reconstruction term  (random t ≥ 2) -------------------------------
    timesteps = jax.random.randint(rng_t, (x0.shape[0],), 2, max_t)
    noise_keys = jax.random.split(rng_t, x0.shape[0])
    x_t = jax.vmap(sched.add_noise)(noise_keys, x0, timesteps)

    x0_pred = jax.vmap(denoiser, in_axes=(0, 0, 0))(x_t,
                                 timesteps.astype(jnp.float32) / max_t,
                                 cond_emb)
    recon = (1. / (max_t - 2)) * jnp.mean(jnp.sum((x0 - x0_pred) ** 2,
                                           axis=(-1, -2)))   # unbiased MC

    # ---- embedding-match  +  NLL  at t = 1 ---------------------------------
    x1        = sched.add_noise(rng_1, x0, 1)
    x0_pred_1 = jax.vmap(denoiser, in_axes=(0, 0, 0))(x1, jnp.ones(shape=(x1.shape[0],)) / max_t, cond_emb)

    embed_err  = jnp.mean(jnp.sum((emb_clean - x0_pred_1) ** 2, axis=(-1, -2)))

    logits = denoiser.embedding.reverse(x0)                      # W x0 + b
    nll    = optax.softmax_cross_entropy_with_integer_labels(logits, tokens).mean()

    # ---- full objective ----------------------------------------------------
    return recon + embed_err + nll + kl_T


