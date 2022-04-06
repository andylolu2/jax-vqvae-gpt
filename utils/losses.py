import jax.numpy as jnp
import jax.nn as nn


def mse(y, y_pred):
    return jnp.mean((y - y_pred) ** 2)


def cross_entropy(y, y_pred):
    return jnp.mean(jnp.sum(-y * nn.log_softmax(y_pred), axis=-1))
