import jax.numpy as jnp
from jax import Array

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
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x_0: Array, t: Array, noise: Array) -> Array:
        """Add noise according to the diffusion forward process.

        This implementation is fully *vmappable*: it does not make any assumption
        about a leading batch dimension.  `t` can be a scalar or an array with
        any shape that broadcasts against the leading dimensions of `x_0` and
        `noise`.
        """
        # Gather the per-t coefficients
        sqrt_alpha_cumprod_t = jnp.take(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alpha_cumprod_t = jnp.take(self.sqrt_one_minus_alphas_cumprod, t)

        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
