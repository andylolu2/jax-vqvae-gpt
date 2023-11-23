from typing import Optional

import haiku as hk
import jax.nn as nn
import jax.numpy as jnp


class CasualSelfAttention(hk.MultiHeadAttention):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is B x N x H
        seq_len = x.shape[1]
        casual_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        # mask is B x num_heads x N x N
        casual_mask = jnp.tile(casual_mask, (x.shape[0], self.num_heads, 1, 1))
        return super().__call__(x, x, x, casual_mask)


class DecoderBlock(hk.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        model_size: int,
        weight_init_scale: float,
        dropout_rate: float,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.casual_atten = CasualSelfAttention(
            num_heads, hidden_dim, weight_init_scale
        )
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        # Structured according to original paper
        # https://arxiv.org/pdf/1706.03762.pdf#section.3
        res = self.casual_atten(x)
        if is_training:
            res = hk.dropout(hk.next_rng_key(), self.dropout_rate, res)
        x += res
        x = hk.LayerNorm(-1, True, True)(x)

        dim = x.shape[-1]
        res = hk.Linear(dim * 4)(x)
        res = nn.gelu(res)
        res = hk.Linear(dim)(res)
        if is_training:
            res = hk.dropout(hk.next_rng_key(), self.dropout_rate, res)
        x += res
        x = hk.LayerNorm(-1, True, True)(x)
        return x


class GPTLmHeadModel(hk.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout_rate: float,
        max_length: int,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        self.model_size = self.num_heads * self.hidden_dim

        self.init_scale = 2.0 / num_layers
        self.embed = hk.Embed(num_classes, self.model_size)
        self.positional_embeddings = hk.get_parameter(
            "pos_embs",
            [self.max_length, self.model_size],
            init=hk.initializers.TruncatedNormal(stddev=0.02),
        )
        self.blocks = [
            DecoderBlock(
                self.num_heads,
                self.hidden_dim,
                self.model_size,
                self.init_scale,
                self.dropout_rate,
            )
            for _ in range(num_layers)
        ]
        self.lm_head = hk.Linear(num_classes)

    def __call__(self, x, is_training: bool):
        seq_length = x.shape[1]
        x = self.embed(x) + self.positional_embeddings[:seq_length]
        for block in self.blocks:
            x = block(x, is_training)
        x = self.lm_head(x)
        # softmax is taken outside
        return x
