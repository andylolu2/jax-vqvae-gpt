from typing import Any, Callable, NamedTuple, Optional
import functools

import haiku as hk
import jax.numpy as jnp
import jax
import optax
from jax._src.random import KeyArray
from optax._src.base import GradientTransformation

from models import CnnEncoder, CnnDecoder, QuantizedCodebook
from utils.annotations import VqVaeBatch, VqVaeState
from utils.losses import mse


class VqVaeApply(NamedTuple):
    encode: Callable[..., Any]
    decode: Callable[..., Any]
    quantize: Callable[..., Any]
    embed: Callable[..., Any]


class VqVaeTrainer:
    def __init__(
        self,
        K: int,
        D: int,
        compression_level: int,
        res_layers: int,
        commitment_loss: float,
        optimizer: Optional[GradientTransformation],
    ):
        self.K = K
        self.D = D
        self.compression_level = compression_level
        self.res_layers = res_layers

        transformed = self.build(
            self.K, self.D, self.compression_level, self.res_layers, commitment_loss
        )
        self.init = transformed.init
        self.apply = VqVaeApply(*transformed.apply)

        self.optimizer = optimizer

    @staticmethod
    def build(
        K: int, D: int, compression_level: int, res_layers: int, commitment_loss: float
    ):
        def f():
            encoder = CnnEncoder(
                out_channels=D,
                downscale_level=compression_level,
                res_layers=res_layers,
                name="encoder",
            )
            decoder = CnnDecoder(
                in_channels=D,
                upscale_level=compression_level,
                res_layers=res_layers,
                name="decoder",
            )
            quantizer = QuantizedCodebook(K, D, commitment_loss, name="quantizer")

            def encode(x, is_training: bool):
                return encoder(x, is_training)

            def decode(x, is_training: bool):
                return decoder(x, is_training)

            def quantize(x):
                return quantizer(x)

            def embed(x):
                return quantizer.embed(x)

            def init(x, is_training: bool):
                encodings = encode(x, is_training)
                result = quantize(encodings)
                x_pred = decode(result["quantize"], is_training)
                z_q = embed(result["encoding_indices"])
                return x_pred, z_q

            return init, (encode, decode, quantize, embed)

        return hk.multi_transform_with_state(f)

    def initial_state(self, rng: KeyArray, batch: VqVaeBatch) -> VqVaeState:
        params, state = self.init(rng, batch["image"], is_training=True)
        opt_state = self.optimizer.init(params)
        return VqVaeState(params, state, opt_state)

    def forward(self, params: hk.Params, state: hk.State, x, is_training: bool):
        z_e, state = self.apply.encode(params, state, None, x, is_training)
        result, state = self.apply.quantize(params, state, None, z_e)
        z_q = result["quantize"]
        x_pred, state = self.apply.decode(params, state, None, z_q, is_training)
        result["x_pred"] = x_pred
        return result, state

    def loss(
        self, params: hk.Params, state: hk.State, batch: VqVaeBatch, is_training: bool
    ):
        x = batch["image"]
        result, state = self.forward(params, state, x, is_training)
        reconstruct_loss = mse(x, result["x_pred"])
        loss = reconstruct_loss + result["codebook_loss"]
        return loss, (state, result)

    @functools.partial(jax.jit, static_argnums=0)
    def update(
        self, vqvae_state: VqVaeState, batch: VqVaeBatch
    ) -> tuple[VqVaeState, dict[str, Any]]:
        assert self.optimizer is not None

        loss_and_grad = jax.value_and_grad(self.loss, has_aux=True)
        (loss, (state, _)), grads = loss_and_grad(
            vqvae_state.params, vqvae_state.state, batch, True
        )
        updates, opt_state = self.optimizer.update(
            grads, vqvae_state.opt_state, vqvae_state.params
        )
        params = optax.apply_updates(vqvae_state.params, updates)
        new_vqvae_state = VqVaeState(params, state, opt_state)
        logs = {"scalar_loss": jax.device_get(loss)}
        return new_vqvae_state, logs

    @functools.partial(jax.jit, static_argnums=0)
    def evaluate(self, vqvae_state: VqVaeState, batch: VqVaeBatch) -> dict[str, Any]:
        loss, (_, result) = self.loss(
            vqvae_state.params, vqvae_state.state, batch, is_training=False
        )
        logs = {
            "scalar_loss": jax.device_get(loss),
            "images_original": batch["image"],
            "images_reconstruction": result["x_pred"],
        }
        return logs

    def lookup_indices(self, vqvae_state: VqVaeState, indices) -> jnp.ndarray:
        z_q, _ = self.apply.embed(vqvae_state.params, vqvae_state.state, None, indices)
        return z_q
