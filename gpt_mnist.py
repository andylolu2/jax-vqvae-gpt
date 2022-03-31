import json

import jax
import jax.nn as nn
import jax.numpy as jnp
import haiku as hk
import optax
from datasets.load import load_from_disk

from models.gpt import build_model
from annotations import GPTConfig, VqVaeConfig, GPTBatch, GPTTrainState

config = GPTConfig(
    seed=23,
    dataset="datasets/exp13-encode",
    num_heads=4,
    hidden_dim=32,
    num_layers=3,
    dropout_rate=0.05,
    vqvae_config="runs/exp13/config.json",
    train_batch_size=32,
    test_batch_size=32,
    learning_rate=3e-4,
    weight_decay=1e-5
)
with open(config.vqvae_config, "r") as f:
    vqvae_config = VqVaeConfig(**json.load(f))


mnist_train = load_from_disk(config.dataset)

model = build_model(config.num_heads,
                    config.hidden_dim,
                    config.num_layers,
                    vqvae_config.K,
                    config.dropout_rate)

key = jax.random.PRNGKey(config.seed)


def cross_entropy(y, y_pred):
    return jnp.sum(-y * nn.log_softmax(y_pred), axis=-1)


def loss(params: hk.Params, state: hk.State, rng, batch: GPTBatch):
    tokens = batch["encoding_indices"]
    tokens = tokens.reshape((tokens.shape[0], -1))
    y_pred, new_state = model.apply(
        params, state, rng, tokens, is_training=True)
    loss = jnp.mean(cross_entropy(tokens[:, 1:], y_pred[:, :-1]))
    return loss, new_state


loss_and_grad = jax.value_and_grad(loss, has_aux=True)
optimizer = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)


@jax.jit
def step(train_state: GPTTrainState, batch: GPTBatch) -> tuple[GPTTrainState, jnp.ndarray]:
    rng, _rng = jax.random.split(train_state.rng)
    (loss_value, state), grads = loss_and_grad(train_state.params,
                                               train_state.state,
                                               _rng,
                                               batch)
    updates, opt_state = optimizer.update(grads,
                                          train_state.opt_state,
                                          train_state.params)
    params = optax.apply_updates(train_state.params, updates)

    train_state = GPTTrainState(params, state, opt_state, rng)
    return train_state, loss_value


@jax.jit
def eval_step(train_state: GPTTrainState, batch: GPTBatch) -> tuple[GPTTrainState, jnp.ndarray]:
    rng, _rng = jax.random.split(train_state.rng)
    loss_val, state = loss(train_state.params, train_state.state, _rng, batch)
    train_state = GPTTrainState(
        train_state.params,
        state,
        train_state.opt_state,
        rng
    )
    return train_state, loss_val
