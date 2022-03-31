from typing import Optional

import haiku as hk
import jax.numpy as jnp
import jax.nn as nn

from annotations import GPTBatch


class CasualSelfAttention(hk.MultiHeadAttention):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # query is B x N x H
        # key is B x N x H
        # value is B x N x H
        seq_len = x.shape[1]
        casual_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        return super().__call__(x, x, x, casual_mask)


class DecoderBlock(hk.Module):
    def __init__(self,
                 num_heads: int,
                 hidden_dim: int,
                 weight_init_scale: float,
                 dropout_rate: float,
                 name: Optional[str] = None):
        super().__init__(name)
        self.casual_atten = CasualSelfAttention(num_heads,
                                                hidden_dim,
                                                weight_init_scale)
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        res = self.casual_atten(x)
        res = hk.dropout(hk.next_rng_key(), self.dropout_rate, res)
        x += res
        x = hk.BatchNorm(True, True, 0.9)(x, is_training)

        dim = x.shape[-1]
        res = hk.Linear(dim)(x)
        res = nn.relu(res)
        x += res
        x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        return x


class GPTLmHeadModel(hk.Module):
    def __init__(self,
                 num_heads: int,
                 hidden_dim: int,
                 num_layers: int,
                 num_classes: int,
                 dropout_rate: float,
                 name: Optional[str] = None):
        super().__init__(name)
        init_scale = 2. / num_layers

        self.embed = hk.Embed(num_classes, hidden_dim)
        self.blocks = [
            DecoderBlock(num_heads, hidden_dim, init_scale, dropout_rate)
            for _ in range(num_layers)]
        self.lm_head = hk.Linear(num_classes)

    def __call__(self, x, is_training: bool):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, is_training)
        x = self.lm_head(x)
        x = nn.softmax(x, axis=-1)
        return x


def build_model(num_heads: int,
                hidden_dim: int,
                num_layers: int,
                num_classes: int,
                dropout_rate: float):
    @hk.transform_with_state
    def model(tokens, is_training: bool):
        # tokens shape B x (W * H)
        net = GPTLmHeadModel(num_heads, hidden_dim,
                             num_layers, num_classes, dropout_rate)
        return net(tokens, is_training)
    return model
