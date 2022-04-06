from typing import Any, NamedTuple, TypedDict

import numpy as np
import haiku as hk
import optax
from jax._src.random import KeyArray


class VqVaeConfig(NamedTuple):
    seed: int
    dataset: str
    resize_shape: tuple[int, int]
    K: int
    D: int
    compression_level: int
    res_layers: int
    commitment_loss: float
    train_dset_percentage: int
    test_dset_percentage: int
    train_steps: int
    test_steps: int
    test_every: int
    train_batch_size: int
    test_batch_size: int
    learning_rate: float
    weight_decay: float
    logdir: str
    output_name: str


class GPTConfig(NamedTuple):
    seed: int
    num_heads: int
    hidden_dim: int
    num_layers: int
    dropout_rate: float
    vqvae_config: str
    vqvae_state: str
    train_steps: int
    test_steps: int
    test_every: int
    train_dataset: str
    test_dataset: str
    train_batch_size: int
    test_batch_size: int
    generate_samples: int
    sample_temperature: float
    learning_rate: float
    weight_decay: float
    logdir: str
    output_name: str


class VqVaeTuple(NamedTuple):
    encoder: Any
    decoder: Any
    quantizer: Any


class VqVaeState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState


class GPTState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState
    rng: KeyArray


class VqVaeBatch(TypedDict):
    image: np.ndarray
    label: np.ndarray


class GPTBatch(TypedDict):
    label: np.ndarray
    encoding_indices: np.ndarray
