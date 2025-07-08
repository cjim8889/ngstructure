from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array


class Embedding(eqx.Module):
    # Parameters
    embedding: eqx.nn.Embedding               # [vocab, d]
    bias: jnp.ndarray                         # [vocab]
    # Static field
    embedding_size: int = eqx.field(static=True)

    # ------------- constructor ------------------------------------------------
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        key: jax.random.PRNGKey,
        *,
        bias_init: float = 0.0
    ):
        key_emb, key_bias = jax.random.split(key)
        self.embedding_size = embedding_size
        self.embedding = eqx.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_size=embedding_size,
            key=key_emb,
        )
        # Bias for the soft-max classifier (optional but useful)
        self.bias = jnp.full((vocab_size,), bias_init)

    # ------------- forward:   tokens -> x0  -----------------------------------
    def forward(
        self,
        tokens: Int[Array, "max_length"],
        noise_std: float = 0.0,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "max_length embedding_size"]:
        """
        Look up embeddings and (optionally) add N(0, noise_std² I).

        Args
        ----
        tokens : int[...]        integer token ids
        noise_std : float        σ₀ in Diffusion-LM; 0.0 means no noise
        key : PRNGKey            required if noise_std > 0
        """
        x0 = jax.vmap(self.embedding)(tokens)                     # shape [max_length embedding_size]
        eps = jax.random.normal(key, x0.shape) * noise_std
        x0_noised = x0 + eps
        return x0, x0_noised

    # Make __call__ an alias for forward so existing code still works.
    __call__ = forward

    def lookup(self, tokens: Int[Array, "max_length"]) -> Float[Array, "max_length embedding_size"]:
        return jax.vmap(self.embedding)(tokens)

    # ------------- reverse:   x0 -> logits  -----------------------------------
    def reverse(self, x: Float[Array, "max_length embedding_size"]) -> Float[Array, "max_length vocab_size"]:
        """
        Compute tied-weight logits pθ(w | x).

        Args
        ----
        x : float[..., d]   a batch of embedding-space vectors

        Returns
        -------
        logits : float[..., vocab]
        """
        # einsum does (batch..., d) · (vocab, d)^T  -> (batch..., vocab)
        logits = jnp.einsum("...d,vd->...v", x, self.embedding.weight) + self.bias
        return logits

    # ------------- helpers -----------------------------------------------------
    def predict(self, x: Float[Array, "max_length embedding_size"]) -> Int[Array, "max_length"]:
        """arg-max decoding; returns integer tokens"""
        return jnp.argmax(self.reverse(x), axis=-1)



if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    embedding = Embedding(vocab_size=100, embedding_size=128, key=key)
    x = jnp.array([0, 1, 2, 3, 4])  # A single datapoint consisting of 5 token IDs
    x_embed = embedding(x)

    x_logits = embedding.reverse(x_embed)
    x_pred = embedding.predict(x_embed)

    print(x_logits.shape)
    print(x_pred.shape)
    print(x_embed.shape)