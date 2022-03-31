from typing import NamedTuple, TypedDict

import numpy as np
import haiku as hk
import optax
from jax._src.random import KeyArray


class VqVaeConfig(NamedTuple):
    seed: int
    K: int
    D: int
    train_dset_percentage: int
    test_dset_percentage: int
    train_steps: int
    test_steps: int
    log_every: int
    train_batch_size: int
    test_batch_size: int
    learning_rate: float
    weight_decay: float
    logdir: str
    output_name: str


class GPTConfig(NamedTuple):
    seed: int
    dataset: str
    num_heads: int
    hidden_dim: int
    num_layers: int
    dropout_rate: float
    vqvae_config: str
    train_batch_size: int
    test_batch_size: int
    learning_rate: float
    weight_decay: float


class TrainState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState


class GPTTrainState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState
    rng: KeyArray


class Batch(TypedDict):
    image: np.ndarray
    label: np.ndarray


class GPTBatch(TypedDict):
    encoding_indices: np.ndarray
