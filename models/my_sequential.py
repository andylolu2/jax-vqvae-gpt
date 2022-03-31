from typing import Callable, Iterable, Optional, Any

import jax.numpy as jnp
import haiku as hk


class MySequential(hk.Module):
    def __init__(self,
                 layers: Iterable[Callable[..., Any]],
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.layers = layers

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x
