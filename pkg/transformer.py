from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
from jaxtyping import Array, Float

from pkg.embedding import Embedding

from .utils import init_linear_weights, xavier_init, zero_init


# Helper modulation function (same as DiT's modulate)
def modulate(x: Float[Array, " ... "], shift: Float[Array, " ... "], scale: Float[Array, " ... "]) -> Float[Array, " ... "]:
    return x * (1 + scale) + shift

class TimeConditionalEmbedding(eqx.Module):
    net: eqx.nn.MLP
    frequency_embedding_size: int = eqx.field(static=True)
    dreams_embedding_size: int = eqx.field(static=True)
    mp_policy: jmp.Policy = eqx.field(static=True)

    def __init__(self, 
            embedding_size: int, 
            frequency_embedding_size: int,
            dreams_embedding_size: int,
            key: jax.random.PRNGKey, 
            mp_policy: jmp.Policy,
            embedder_width: int = 128, 
            embedder_depth: int = 3,
        ):
        self.frequency_embedding_size = frequency_embedding_size
        self.dreams_embedding_size = dreams_embedding_size
        self.mp_policy = mp_policy
        self.net = eqx.nn.MLP(
            in_size=frequency_embedding_size + dreams_embedding_size,
            out_size=embedding_size,
            width_size=embedder_width,
            depth=embedder_depth,
            activation=jax.nn.silu,
            use_bias=True,
            key=key,
        )

    @staticmethod
    def timestep_embedding(
        t: Float[Array, "..."], 
        dim: int, 
        max_period: int = 10000
    ) -> Float[Array, "... dim"]:
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: A 1-D array of N indices (may be fractional).
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.

        Returns:
            A (N, dim) array of positional embeddings.
        """
        half = dim // 2
        # Compute frequencies.
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half
        )
        # Ensure t is a scalar float32.
        t = jnp.asarray(t, dtype=jnp.float32)
        args = t * freqs
        # Concatenate cosine and sine embeddings.
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        # Pad with an extra zero if dim is odd.
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[..., :1])], axis=-1)
        return embedding
    
    def __call__(self, t: Float[Array, ""], dreams_embeddings: Float[Array, "dreams_embedding_size"]) -> Float[Array, "..."]:
        # Compute the sinusoidal embeddings.
        t_freq = TimeConditionalEmbedding.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = jnp.concatenate([t_freq, dreams_embeddings], axis=-1)

        t_freq = self.mp_policy.cast_to_compute(t_freq)
        net = self.mp_policy.cast_to_compute(self.net)
        
        # Pass the embeddings through the MLP.
        t_emb = net(t_freq)
        t_emb = self.mp_policy.cast_to_output(t_emb)
        return t_emb
    
class EfficientFFN(eqx.Module):
    """Optimized feed-forward network with parameter reuse."""
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    mp_policy: jmp.Policy = eqx.field(static=True)

    def __init__(self, input_size: int, hidden_size: int, key: jax.random.PRNGKey, mp_policy: jmp.Policy):
        self.mp_policy = mp_policy
        key1, key2 = jax.random.split(key, 2)
        self.linear1 = eqx.nn.Linear(input_size, hidden_size, key=key1, dtype=mp_policy.param_dtype)
        self.linear2 = eqx.nn.Linear(hidden_size, input_size, key=key2, dtype=mp_policy.param_dtype)

    def __call__(
        self, 
        x: Float[Array, "max_length embedding_size"],
    ) -> Float[Array, "max_length embedding_size"]:
        x = self.mp_policy.cast_to_compute(x)
        linear1 = self.mp_policy.cast_to_compute(self.linear1)
        linear2 = self.mp_policy.cast_to_compute(self.linear2)
        
        residual = x
        x = jax.vmap(linear1)(x)
        x = jax.nn.silu(self.mp_policy.cast_to_param(x))
        x = jax.vmap(linear2)(x) + residual
        x = self.mp_policy.cast_to_output(x)

        return x

###############################################################################
#      Adaptive LayerNorm Modulation (produces 6 parameters per block)       #
###############################################################################
class AdaptiveLayerNormModulation(eqx.Module):
    linear: eqx.nn.Linear
    count: int = eqx.field(static=True)
    mp_policy: jmp.Policy = eqx.field(static=True)

    def __init__(self, embedding_size: int, count: int, key: jax.random.PRNGKey, mp_policy: jmp.Policy):
        # Mimic: SiLU -> Linear(embedding_size, count*embedding_size)
        self.count = count
        self.linear = eqx.nn.Linear(embedding_size, count * embedding_size, key=key, dtype=mp_policy.param_dtype)
        self.mp_policy = mp_policy

        # Initialize weights to zero
        self.linear = init_linear_weights(self.linear, xavier_init, key=key)

    def __call__(self, c: Float[Array, "embedding_size"]) -> List[Float[Array, "embedding_size"]]:
        # c: conditioning vector (shape: (hidden_size,) or (batch, hidden_size))
        c = jax.nn.silu(c)
        c = self.mp_policy.cast_to_compute(c)
        linear = self.mp_policy.cast_to_compute(self.linear)
        params = self.mp_policy.cast_to_output(linear(c))  # shape: (count*hidden_size,)
        return jnp.split(params, self.count, axis=-1)  # returns [shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn]



class DiTBlock(eqx.Module):
    layernorm1: eqx.nn.LayerNorm
    layernorm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    rotary_embeddings: eqx.nn.RotaryPositionalEmbedding
    modulation: AdaptiveLayerNormModulation
    ffn: EfficientFFN
    mp_policy: jmp.Policy = eqx.field(static=True)

    def __init__(self, embedding_size: int, hidden_size: int, num_heads: int, key: jax.random.PRNGKey, mp_policy: jmp.Policy):
        self.mp_policy = mp_policy
        self.layernorm1 = eqx.nn.LayerNorm(embedding_size, dtype=self.mp_policy.param_dtype)
        self.layernorm2 = eqx.nn.LayerNorm(embedding_size, dtype=self.mp_policy.param_dtype)

        key1, key2, key3 = jax.random.split(key, 3)
        
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embedding_size,
            key=key1,
            dtype=self.mp_policy.param_dtype,
        )
        
        self.ffn = EfficientFFN(
            input_size=embedding_size,
            hidden_size=hidden_size,
            key=key2,
            mp_policy=mp_policy,
        )

        self.rotary_embeddings = eqx.nn.RotaryPositionalEmbedding(
            embedding_size=embedding_size // num_heads,
            theta=10000.0,
            dtype=self.mp_policy.param_dtype,
        )

        self.modulation = AdaptiveLayerNormModulation(embedding_size, count=6, key=key3, mp_policy=mp_policy)


    def __call__(self, 
            x: Float[Array, "max_length embedding_size"],
            c: Float[Array, "embedding_size"],
        ) -> Float[Array, "max_length embedding_size"]:
        def process_heads(
            query_heads: Float[Array, "max_length num_heads qk_size"],
            key_heads: Float[Array, "max_length num_heads qk_size"],
            value_heads: Float[Array, "max_length num_heads vo_size"]
        ) -> tuple[Float[Array, "max_length num_heads qk_size"], 
                  Float[Array, "max_length num_heads qk_size"], 
                  Float[Array, "max_length num_heads vo_size"]]:
            query_heads = jax.vmap(self.rotary_embeddings, in_axes=1, out_axes=1)(query_heads)
            key_heads = jax.vmap(self.rotary_embeddings, in_axes=1, out_axes=1)(key_heads)
            return query_heads, key_heads, value_heads
        
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.modulation(c)
        # Expand to per-token shape:
        shift_attn = jnp.broadcast_to(shift_attn, x.shape)
        scale_attn = jnp.broadcast_to(scale_attn, x.shape)
        gate_attn  = jnp.broadcast_to(gate_attn, x.shape)
        shift_ffn  = jnp.broadcast_to(shift_ffn, x.shape)
        scale_ffn  = jnp.broadcast_to(scale_ffn, x.shape)
        gate_ffn   = jnp.broadcast_to(gate_ffn, x.shape)


        # Attention branch:
        x_norm_attn = jax.vmap(self.layernorm1)(x)
        x_mod_attn = modulate(x_norm_attn, shift_attn, scale_attn)
        attn_out = self.attention(
            query=x_mod_attn,
            key_=x_mod_attn,
            value=x_mod_attn,
            inference=True,
            process_heads=process_heads,
        )
        x = x + gate_attn * attn_out

        # FFN branch:
        x_norm_ffn = jax.vmap(self.layernorm2)(x)
        x_mod_ffn = modulate(x_norm_ffn, shift_ffn, scale_ffn)
        ffn_out = self.ffn(x_mod_ffn)
        x = x + gate_ffn * ffn_out

        return x
    
###############################################################################
#                   DiT-Style Transformer Layer                           #
###############################################################################
class FinalLayer(eqx.Module):
    """
    The final layer of DiT.
    """
    norm_final: eqx.nn.LayerNorm
    linear: eqx.nn.Linear
    adaLN_modulation: AdaptiveLayerNormModulation
    mp_policy: jmp.Policy = eqx.field(static=True)
    
    def __init__(self, embedding_size: int, output_size: int, key: jax.random.PRNGKey, mp_policy: jmp.Policy):
        self.norm_final = eqx.nn.LayerNorm(embedding_size, use_bias=False, use_weight=False, eps=1e-6)
        key1, key2 = jax.random.split(key)
        self.linear = eqx.nn.Linear(embedding_size, output_size, key=key2, dtype=mp_policy.param_dtype)
        self.adaLN_modulation = AdaptiveLayerNormModulation(embedding_size, count=2, key=key1, mp_policy=mp_policy)
        # Initialize weights to zero
        self.linear = init_linear_weights(self.linear, zero_init, key=key2)

        self.mp_policy = mp_policy

    def __call__(self, x: Float[Array, "max_length embedding_size"], c: Float[Array, "embedding_size"]) -> Float[Array, "max_length output_size"]:
        shift, scale = self.adaLN_modulation(c)
        # Expand to per-token shape:
        shift = jnp.broadcast_to(shift, x.shape)
        scale = jnp.broadcast_to(scale, x.shape)

        x = modulate(jax.vmap(self.norm_final)(x), shift, scale)

        x = self.mp_policy.cast_to_compute(x)
        linear = self.mp_policy.cast_to_compute(self.linear)
        x = self.mp_policy.cast_to_output(jax.vmap(linear)(x))
        return x


###############################################################################
#                    Sequence DiT using DiT-style layers                      #
###############################################################################
class SequenceDiT(eqx.Module):
    """
    Sequence Diffusion Transformer with DiT-style adaptive layer norm conditioning.
    Handles input of shape (max_length, embedding_size) without batch dimension.
    """
    time_conditional_embedder: TimeConditionalEmbedding
    layers: List[DiTBlock]
    predictor: FinalLayer
    embedding: Embedding
    mp_policy: jmp.Policy = eqx.field(static=True)
    embedding_size: int = eqx.field(static=True)
    dreams_embedding_size: int = eqx.field(static=True)

    def __init__(
        self,
        max_length: int,
        vocab_size: int,
        embedding_size: int,
        dreams_embedding_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        key: jax.random.PRNGKey,
        mp_policy: jmp.Policy,
        frequency_embedding_size: int = 256,
        embedder_width: int = 128,
        embedder_depth: int = 2,
    ):
        self.mp_policy = mp_policy
        self.embedding_size = embedding_size
        self.dreams_embedding_size = dreams_embedding_size

    
        e_key, l_key, p_key, t_key = jax.random.split(key, 4)

        self.embedding = Embedding(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            key=e_key,
        )

        self.time_conditional_embedder = TimeConditionalEmbedding(
            embedding_size=embedding_size,
            frequency_embedding_size=frequency_embedding_size,
            dreams_embedding_size=dreams_embedding_size,
            key=t_key,
            mp_policy=mp_policy,
            embedder_width=embedder_width,
            embedder_depth=embedder_depth,
        )

        self.layers = [
            DiTBlock(
                embedding_size=embedding_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                key=k,
                mp_policy=mp_policy,
            )
            for k in jax.random.split(l_key, num_layers)
        ]

        self.predictor = FinalLayer(embedding_size, embedding_size, key=p_key, mp_policy=mp_policy)

    def __call__(
        self,
        tokens: Float[Array, "max_length embedding_size"],
        t: Float[Array, ""],
        dreams_embeddings: Float[Array, "dreams_embedding_size"],
    ) -> Float[Array, "max_length embedding_size"]:

        c = self.time_conditional_embedder(t, dreams_embeddings)  # (embedding_size,)

        tokens = self.mp_policy.cast_to_compute(tokens)
        x = self.mp_policy.cast_to_compute(tokens)

        for layer in self.layers:
            x = layer(x, c)

        return self.predictor(x, c)
    

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    mp_policy = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        output_dtype=jnp.float32,
    )   
    model = SequenceDiT(
        max_length=1024,
        embedding_size=128,
        dreams_embedding_size=128,
        hidden_size=128,
        num_layers=12,
        num_heads=8,
        key=key,
        mp_policy=mp_policy,
        frequency_embedding_size=128,
        embedder_width=128,
        embedder_depth=2,
    )


    tokens = jnp.zeros((1024, 128))
    t = jnp.zeros(())
    dreams_embeddings = jnp.zeros((128,))

    output = model(tokens, t, dreams_embeddings)
    print(output.shape)