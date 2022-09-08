from typing import Optional
import functools

import haiku as hk
import jax.numpy as jnp
import jax.nn as nn
import jax
import optax
from jax._src.random import KeyArray
from optax._src.base import GradientTransformation

from models import GPTLmHeadModel
from utils.annotations import GPTBatch, GPTState, VqVaeConfig
from utils.losses import cross_entropy


class VqVaeGPTTrainer:
    def __init__(
        self,
        num_label_classes: int,
        vqvae_config: VqVaeConfig,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        dropout_rate: float,
        sample: GPTBatch,
        optimizer: Optional[GradientTransformation],
    ):
        self.vqvae_config = vqvae_config
        self.num_label_classes = num_label_classes
        self.num_classes = num_label_classes + vqvae_config.K
        self.decoder_input_shape = sample["encoding_indices"].shape[1:]
        self.seq_length = self.tokenize(sample).shape[-1]

        transformed = self.build(
            num_heads,
            hidden_dim,
            num_layers,
            self.num_classes,
            dropout_rate,
            self.seq_length,
        )
        self.init = transformed.init
        self.apply = transformed.apply

        self.optimizer = optimizer

    @staticmethod
    def build(
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout_rate: float,
        seq_length: int,
    ):
        def init(tokens, is_training: bool):
            net = GPTLmHeadModel(
                num_heads, hidden_dim, num_layers, num_classes, dropout_rate, seq_length
            )
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

    def forward(
        self, params: hk.Params, state: hk.State, rng, tokens, is_training: bool
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
        updates, opt_state = self.optimizer.update(
            grads, gpt_state.opt_state, gpt_state.params
        )
        params = optax.apply_updates(gpt_state.params, updates)

        new_gpt_state = GPTState(params, state, opt_state, rng1)
        logs = {"scalar_loss": jax.device_get(loss)}
        return new_gpt_state, logs

    @functools.partial(jax.jit, static_argnums=0)
    def evaluate(self, gpt_state: GPTState, batch: GPTBatch) -> tuple[GPTState, dict]:
        tokens = self.tokenize(batch)
        loss, state = self.loss(gpt_state.params, gpt_state.state, None, tokens, False)
        new_gpt_state = GPTState(
            gpt_state.params, state, gpt_state.opt_state, gpt_state.rng
        )
        logs = {
            "scalar_loss": jax.device_get(loss),
        }
        return new_gpt_state, logs

    @functools.partial(jax.jit, static_argnums=0)
    def generate(self, gpt_state: GPTState, rng: KeyArray, label: int, temp: float = 1):
        output_len = self.seq_length - 1
        padded_tokens = [[label] + [0] * output_len]
        tokens = jnp.array(padded_tokens, dtype=jnp.int32)

        def body_fun(i, val: tuple[jnp.ndarray, GPTState, KeyArray]):
            # token shape 1 x (W * H + 1)
            tokens, gpt_state, rng = val

            y_pred, _ = self.apply(
                gpt_state.params, gpt_state.state, None, tokens, False
            )
            probs = (y_pred[0, i, :] / temp)[self.num_label_classes :]
            probs = nn.softmax(probs)

            vqvae_tokens = jnp.arange(
                self.num_label_classes, self.num_classes, dtype=jnp.int32
            )
            rng, rng1 = jax.random.split(rng)
            next_token = jax.random.choice(rng1, vqvae_tokens, p=probs)
            tokens = tokens.at[0, i + 1].set(next_token)

            return (tokens, gpt_state, rng)

        # token shape 1 x (H * W + 1)
        tokens, _, rng = jax.lax.fori_loop(
            0, output_len, body_fun, (tokens, gpt_state, rng)
        )
        # shape 1 x H * W
        tokens = tokens[0, 1:]
        # shape 1 x H x W
        tokens = jnp.reshape(tokens, self.decoder_input_shape)[None, ...]
        tokens = tokens - self.num_label_classes
        return tokens, rng
