import jax.numpy as jnp
from jax import Array
import jax

class DiffusionScheduler:
    """Simple linear noise scheduler for diffusion training."""
    
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.num_timesteps = num_timesteps
        
        # Log-linear noise schedule (betas spaced evenly in log-space)
        self.betas = jnp.exp(
            jnp.linspace(jnp.log(beta_start), jnp.log(beta_end), num_timesteps)
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_alpha_cumprod_0 = self.sqrt_alphas_cumprod[0]
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)

    def posterior_mean(
        self,
        x_t: Array,        # noisy sample at step t   (… × d)
        x_0: Array,        # clean sample            (… × d)
        t:  Array | int,   # int or array broadcastable to x_t.shape[:-1]
    ) -> Array:
        """
        Return μ̂_t(x_t , x_0)  =  c₁(t) · x₀  +  c₂(t) · x_t
        with   c₁ = √ā_{t-1}·β_t / (1−ā_t),
               c₂ = √α_t·(1−ā_{t-1}) / (1−ā_t).

        **t must be ≥ 1.**  (At t=0 the posterior is degenerate.)
        """
        t = jnp.asarray(t)

        beta_t   = jnp.take(self.betas, t)                # β_t
        alpha_t  = 1.0 - beta_t                           # α_t
        bar_alpha_t   = jnp.take(self.alphas_cumprod, t)  # ā_t
        bar_alpha_tm1 = jnp.take(self.alphas_cumprod, t - 1)  # ā_{t-1}

        c1 = (jnp.sqrt(bar_alpha_tm1) * beta_t) / (1.0 - bar_alpha_t)
        c2 = (jnp.sqrt(alpha_t) * (1.0 - bar_alpha_tm1)) / (1.0 - bar_alpha_t)

        return c1 * x_0 + c2 * x_t
    
    def add_noise(self, key: jax.random.PRNGKey, x_0: Array, t: Array) -> Array:
        """Add noise according to the diffusion forward process.

        This implementation is fully *vmappable*: it does not make any assumption
        about a leading batch dimension.  `t` can be a scalar or an array with
        any shape that broadcasts against the leading dimensions of `x_0` and
        `noise`.
        """
        noise = jax.random.normal(key, x_0.shape)
        # Gather the per-t coefficients
        sqrt_alpha_cumprod_t = jnp.take(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alpha_cumprod_t = jnp.take(self.sqrt_one_minus_alphas_cumprod, t)

        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
