from typing import Optional
import functools

import haiku as hk
import jax.numpy as jnp
import jax.nn as nn
import jax
import optax
from optax._src.base import GradientTransformation

from annotations import GPTBatch, GPTState, VqVaeConfig
from losses import cross_entropy


class CasualSelfAttention(hk.MultiHeadAttention):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is B x N x H
        seq_len = x.shape[1]
        # jnp.tril([[1,1,1],[1,1,1],[1,1,1]])
        # = [[1,0,0],[1,1,0],[1,1,1]]
        casual_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        # mask shape is B x num_heads x N x N
        casual_mask = jnp.tile(casual_mask, (x.shape[0], self.num_heads, 1, 1))
        return super().__call__(x, x, x, casual_mask)


class DecoderBlock(hk.Module):
    def __init__(self,
                 num_heads: int,
                 hidden_dim: int,
                 model_size: int,
                 weight_init_scale: float,
                 dropout_rate: float,
                 name: Optional[str] = None):
        super().__init__(name)
        self.casual_atten = CasualSelfAttention(num_heads,
                                                hidden_dim,
                                                weight_init_scale,
                                                model_size)
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        res = self.casual_atten(x)
        if is_training:
            res = hk.dropout(hk.next_rng_key(), self.dropout_rate, res)
        x += res
        x = hk.LayerNorm(-1, True, True)(x)

        dim = x.shape[-1]
        res = hk.Linear(dim * 4)(x)
        res = nn.gelu(res)
        res = hk.Linear(dim)(res)
        x += res
        x = hk.LayerNorm(-1, True, True)(x)
        return x


class GPTLmHeadModel(hk.Module):
    def __init__(self,
                 num_heads: int,
                 hidden_dim: int,
                 num_layers: int,
                 num_classes: int,
                 dropout_rate: float,
                 max_length: int,
                 name: Optional[str] = None):
        super().__init__(name)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        self.model_size = self.num_heads * self.hidden_dim

        self.init_scale = 2. / num_layers
        self.embed = hk.Embed(num_classes, self.model_size)
        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        self.positional_embeddings = hk.get_parameter(
            'pos_embs', [self.max_length, self.model_size], init=embed_init
        )
        self.blocks = [
            DecoderBlock(self.num_heads,
                         self.hidden_dim,
                         self.model_size,
                         self.init_scale,
                         self.dropout_rate)
            for _ in range(num_layers)]
        self.lm_head = hk.Linear(num_classes)

    def __call__(self, x, is_training: bool):
        seq_length = x.shape[1]
        x = self.embed(x) + self.positional_embeddings[:seq_length]
        for block in self.blocks:
            x = block(x, is_training)
        x = self.lm_head(x)
        return x


class VqVaeGPTModel:
    def __init__(self,
                 num_label_classes: int,
                 vqvae_config: VqVaeConfig,
                 num_heads: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout_rate: float,
                 sample: GPTBatch,
                 optimizer: Optional[GradientTransformation]):
        self.vqvae_config = vqvae_config
        self.num_label_classes = num_label_classes
        self.num_classes = num_label_classes + vqvae_config.K
        self.decoder_input_shape = sample["encoding_indices"].shape[1:]
        self.seq_length = self.tokenize(sample).shape[-1]

        transformed = self.build(num_heads,
                                 hidden_dim,
                                 num_layers,
                                 self.num_classes,
                                 dropout_rate,
                                 self.seq_length)
        self.init = transformed.init
        self.apply = transformed.apply

        self.optimizer = optimizer

    @staticmethod
    def build(num_heads: int,
              hidden_dim: int,
              num_layers: int,
              num_classes: int,
              dropout_rate: float,
              seq_length: int):
        def init(tokens, is_training: bool):
            net = GPTLmHeadModel(num_heads,
                                 hidden_dim,
                                 num_layers,
                                 num_classes,
                                 dropout_rate,
                                 seq_length)
            return net(tokens, is_training)
        return hk.transform_with_state(init)

    def initial_state(self, rng, batch: GPTBatch) -> GPTState:
        tokens = self.tokenize(batch)

        rng, rng1 = jax.random.split(rng)
        params, state = self.init(rng, tokens, is_training=True)
        opt_state = self.optimizer.init(params)

        return GPTState(params, state, opt_state, rng1)

    def tokenize(self, batch: GPTBatch):
        # labels shape B x 1
        labels = batch["label"][..., None]
        vqvae_tokens = batch["encoding_indices"]
        # tokens shape B x (W * H)
        vqvae_tokens = vqvae_tokens.reshape((vqvae_tokens.shape[0], -1))
        # offset encoding indices
        vqvae_tokens += self.num_label_classes

        # add labels as additional tokens
        # tokens shape B x (1 + W * H)
        tokens = jnp.concatenate((labels, vqvae_tokens), axis=-1)
        return tokens

    def forward(self, params: hk.Params, state: hk.State, rng, tokens, is_training: bool
                ) -> tuple[jnp.ndarray, hk.State]:
        y_pred, state = self.apply(params, state, rng, tokens, is_training)
        return y_pred, state

    def loss(self, params: hk.Params, state: hk.State, rng, tokens, is_training: bool):
        y = nn.one_hot(tokens, self.num_classes)
        y_pred, state = self.forward(params, state, rng, tokens, is_training)

        # use the first n-1 tokens to predict the nth token
        loss = cross_entropy(y[:, 1:], y_pred[:, :-1])
        return loss, state

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, gpt_state: GPTState, batch: GPTBatch) -> tuple[GPTState, dict]:
        assert self.optimizer is not None

        rng, rng1 = jax.random.split(gpt_state.rng)
        tokens = self.tokenize(batch)
        loss_and_grad = jax.value_and_grad(self.loss, has_aux=True)
        (loss, state), grads = loss_and_grad(
            gpt_state.params, gpt_state.state, rng, tokens, True
        )
        updates, opt_state = self.optimizer.update(grads,
                                                   gpt_state.opt_state,
                                                   gpt_state.params)
        params = optax.apply_updates(gpt_state.params, updates)

        new_gpt_state = GPTState(params, state, opt_state, rng1)
        logs = {
            "scalar_loss": jax.device_get(loss)
        }
        return new_gpt_state, logs

    @functools.partial(jax.jit, static_argnums=0)
    def evaluate(self, gpt_state: GPTState, batch: GPTBatch) -> tuple[GPTState, dict]:
        rng, rng1 = jax.random.split(gpt_state.rng)
        tokens = self.tokenize(batch)
        loss, state = self.loss(
            gpt_state.params, gpt_state.state, rng, tokens, False)
        new_gpt_state = GPTState(
            gpt_state.params, state, gpt_state.opt_state, rng1)
        logs = {
            "scalar_loss": jax.device_get(loss),
        }
        return new_gpt_state, logs

    @functools.partial(jax.jit, static_argnums=0)
    def generate(self, gpt_state: GPTState, label: int):
        tokens = jnp.array([[label]], dtype=jnp.int32)
        output_len = self.seq_length - 1

        rng = gpt_state.rng
        params, state = gpt_state.params, gpt_state.state
        for _ in range(output_len):
            rng, rng1, rng2 = jax.random.split(rng, num=3)
            # y_pred shape 1 x N x (L + K)
            y_pred, state = self.apply(params, state, rng1, tokens, False)
            # next_token shape 1 x 1
            next_token = jnp.argmax(y_pred[:, -1], axis=-1, keepdims=True)
            next_token = jnp.clip(next_token, self.num_label_classes, None)
            # token shape 1 x (N+1)
            tokens = jnp.concatenate((tokens, next_token), axis=-1)
        # shape 1 x N
        tokens = tokens[:, 1:]
        # shape 1 x H x W
        tokens = jnp.reshape(tokens, self.decoder_input_shape)[None, ...]
        tokens = jnp.clip(tokens - self.num_label_classes, 0, None)
        return tokens
