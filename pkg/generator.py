
from typing import List, Optional, Tuple

import equinox as eqx  # ↳ light-weight JAX NN library
import jax
import jax.numpy as jnp

# deepchem's tokenizer still works; we only need it on the host
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from jaxtyping import Array

from pkg.scheduler import DiffusionScheduler
from pkg.transformer import SequenceDiT


class MolecularGenerator:
    # ------------- static / immutable fields --------------------------------
    model:          SequenceDiT              # DiT denoiser  fθ
    scheduler:      "DiffusionScheduler"       # contains β, ᾱ, …
    tokenizer:      SmilesTokenizer
    max_length:     int
    guidance_scale: float                      # (kept for API parity)

    # ------------- pre-computed constant arrays (stay on device) ------------
    token_embeddings:  jnp.ndarray             # [V, d_model]   (ℓ₂-normalised)
    emb_min:           jnp.ndarray             # [1, 1, d_model] coord-wise min
    emb_max:           jnp.ndarray             # [1, 1, d_model] coord-wise max

    # ------------------------------------------------------------------------
    def __init__(
        self,
        model:          SequenceDiT,
        scheduler:      "DiffusionScheduler",
        tokenizer:      SmilesTokenizer,
        *,
        max_length:     int  = 500,
        guidance_scale: float = 1.0,
    ):
        self.model          = model
        self.scheduler      = scheduler
        self.tokenizer      = tokenizer
        self.max_length     = max_length
        self.guidance_scale = guidance_scale

        # ---- cache token embeddings on device --------------------------------
        ids = jnp.arange(tokenizer.vocab_size)
        tok_emb = model.embedding.lookup(ids)                # (V, d)
        tok_emb = tok_emb / jnp.linalg.norm(tok_emb, axis=-1, keepdims=True)
        self.token_embeddings = tok_emb

        emb_min = tok_emb.min(axis=0, keepdims=True)[None, ...]   # (1,1,d)
        emb_max = tok_emb.max(axis=0, keepdims=True)[None, ...]

        self.emb_min = emb_min
        self.emb_max = emb_max


    @eqx.filter_jit
    def _nearest_vocab_embedding(self, x: Array) -> Array:
        """
        Args
        ----
        x : [L, d]  – arbitrary vectors
        Returns
        -------
        x_nn : [L, d]  – each vector replaced by the embedding of the
                            most similar token (cos-sim).
        """
        L, D = x.shape
        x_norm  = x / jnp.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)
        # token_embeddings already ℓ₂-normalised in __init__    
        # sims: [L,V]
        sims = jnp.einsum("ld,vd->lv", x_norm, self.token_embeddings)
        idx  = jnp.argmax(sims, axis=-1)            # [L] int32
        return jnp.take(self.token_embeddings, idx, axis=0) 

    def _emb_to_tokens(self, x: Array) -> Array:
        """Return integer tokens [L] via nearest-neighbour lookup."""
        L, D = x.shape
        x_unit  = x / jnp.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)
        sims    = jnp.einsum("ld,vd->lv", x_unit, self.token_embeddings)
        return jnp.argmax(sims, axis=-1).astype(jnp.int32)

    # ---------- posterior σ_t (scalar) --------------------------------------
    def _posterior_sigma(self, t: int) -> float:
        """
        σ_t = sqrt(β_t * (1-ā_{t-1}) / (1-ā_t))   for  t ≥ 1.
        """
        betas     = self.scheduler.betas
        abar      = self.scheduler.alphas_cumprod
        beta_t    = betas[t]
        abar_t    = abar[t]
        abar_tm1  = jnp.where(t == 0, 1.0, abar[t - 1])
        var       = beta_t * (1.0 - abar_tm1) / (1.0 - abar_t)
        return jnp.where(t == 0, 0.0, jnp.sqrt(var))

    # ---------- single reverse step  t  →  t-1 ------------------------------
    @eqx.filter_jit
    def _reverse_step(
        self,
        key: jax.random.PRNGKey,
        x_t: Array,                  # [L,d]
        eps_theta: Array,            # [L,d]   predicted score
        t: int
    ) -> Array:
        """Implements Eq.(17) of Diffusion-LM with the NN-clamp."""
        sqrt_abar      = self.scheduler.sqrt_alphas_cumprod
        sqrt_one_mabar = self.scheduler.sqrt_one_minus_alphas_cumprod

        # ---- predict x0 -----------------------------------------------------
        coeff1 = 1.0 / sqrt_abar[t]
        coeff2 = sqrt_one_mabar[t] / sqrt_abar[t]
        x0_hat = coeff1 * x_t + coeff2 * eps_theta

        # ---- clamp to nearest token embedding ------------------------------
        x0_hat = self._nearest_vocab_embedding(x0_hat)

        # ---- mean of posterior  q(x_{t-1}|x_t,x0) --------------------------
        mean_prev = self.scheduler.posterior_mean(x_t, x0_hat, t)

        # ---- sample ---------------------------------------------------------
        key, sub = jax.random.split(key)
        noise    = jax.random.normal(sub, x_t.shape)
        sigma    = self._posterior_sigma(t)
        return mean_prev + sigma * noise

    # ---------- last-step clean prediction ----------------------------------
    @eqx.filter_jit
    def _predict_x0(self, x_t: Array, eps_theta: Array, t: int) -> Array:
        sqrt_abar      = self.scheduler.sqrt_alphas_cumprod
        sqrt_one_mabar = self.scheduler.sqrt_one_minus_alphas_cumprod
        x0_hat = (x_t + sqrt_one_mabar[t] * eps_theta) / sqrt_abar[t]
        return self._nearest_vocab_embedding(x0_hat)

    # =======================================================================
    #                         PUBLIC  SAMPLER
    # =======================================================================
    def sample(
        self,
        key: jax.random.PRNGKey,
        dreams_embeddings: Array,      # [cond_dim]
        *,
        temperature: float = 1.0,
        num_sampling_steps: Optional[int] = None
    ) -> Tuple[Array, List[str]]:
        """
        Reverse-diffusion sampling with classifier-free guidance (guidance_scale).

        Returns
        -------
        tokens : [max_length]  JAX int32 array
        smiles : str        decoded on the host
        """
        d_model    = self.model.embedding_size
        T          = self.scheduler.num_timesteps
        S          = num_sampling_steps or T

        # linear schedule T-1 … 0  (int32)
        timesteps = jnp.linspace(T - 1, 0, S, dtype=jnp.int32)

        # initial Gaussian noise
        key, sub = jax.random.split(key)
        x_t = jax.random.normal(sub, (self.max_length, d_model)) * temperature

        # ---- reverse diffusion loop (lax.scan) ------------------------------
        def loop(carry, t):
            key, x_curr = carry
            key, sub = jax.random.split(key)

            # ε_θ(x_t,t,cond) ;  model expects time in [0,1]
            eps_theta = self.model(
                x_curr, jnp.full((1,), t / T), dreams_embeddings
            )

            x_prev = jax.lax.cond(
                t > 0,
                lambda: self._reverse_step(sub, x_curr, eps_theta, t),
                lambda: self._predict_x0(x_curr, eps_theta, t)
            )
            return (key, x_prev), None

        (key, x_final), _ = jax.lax.scan(loop, (key, x_t), timesteps)

        # → tokens
        tokens_jax = self._emb_to_tokens(x_final)
        return tokens_jax
    
    def decode(self, tokens: Array) -> str:
        """Decode tokens to SMILES string."""
        return self.tokenizer.decode(tokens.tolist())



