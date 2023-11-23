from typing import Optional

import haiku as hk
import jax
import jax.nn as nn
import jax.numpy as jnp


class ResBlock(hk.Module):
    def __init__(self, dim: int, kernel_size: int, name: Optional[str] = None):
        super().__init__(name)
        self.dim = dim
        self.kernel_size = kernel_size

    def __call__(self, x, is_training: bool) -> jnp.ndarray:
        res = hk.Conv2D(self.dim, self.kernel_size)(x)
        res = hk.BatchNorm(True, True, 0.9)(res, is_training)
        res = nn.relu(res)
        res = hk.Conv2D(self.dim, self.kernel_size)(res)
        res = hk.BatchNorm(True, True, 0.9)(res, is_training)
        x += res
        x = nn.relu(x)
        return x


class CnnEncoder(hk.Module):
    def __init__(
        self,
        out_channels: int,
        downscale_level: int,
        res_layers: int = 1,
        kernel_size: int = 5,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.out_channels = out_channels
        self.downscale_level = downscale_level
        self.res_layers = res_layers
        self.kernel_size = kernel_size

    def __call__(self, x, is_training: bool) -> jnp.ndarray:
        for i in range(self.downscale_level - 1, -1, -1):
            num_channels = self.out_channels // (2**i)
            x = hk.Conv2D(num_channels, self.kernel_size, stride=2)(x)
            x = hk.BatchNorm(True, True, 0.9)(x, is_training)
            x = nn.relu(x)
            for _ in range(self.res_layers):
                x = ResBlock(num_channels, self.kernel_size)(x, is_training)
        return x


class CnnDecoder(hk.Module):
    def __init__(
        self,
        in_channels: int,
        upscale_level: int,
        res_layers: int = 1,
        kernel_size: int = 5,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.in_channels = in_channels
        self.upscale_level = upscale_level
        self.res_layers = res_layers
        self.kernel_size = kernel_size

    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        for i in range(self.upscale_level - 1):
            num_channels = self.in_channels // (2**i)
            x = hk.Conv2DTranspose(num_channels, self.kernel_size, stride=2)(x)
            x = hk.BatchNorm(True, True, 0.9)(x, is_training)
            x = nn.relu(x)
            for _ in range(self.res_layers):
                x = ResBlock(num_channels, self.kernel_size)(x, is_training)
        x = hk.Conv2DTranspose(1, self.kernel_size, stride=2)(x)
        x = nn.sigmoid(x)
        return x


class QuantizedCodebook(hk.Module):
    def __init__(
        self,
        embed_size_K: int,
        embed_dim_D: int,
        commitment_loss: float,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.K = embed_size_K
        self.D = embed_dim_D
        self.beta = commitment_loss

        initializer = hk.initializers.VarianceScaling(distribution="uniform")
        self.codebook = hk.get_parameter("codebook", (self.K, self.D), init=initializer)

    def __call__(self, inputs) -> dict[str, jnp.ndarray]:
        """Connects the module to some inputs.

        Args:
            inputs: Tensor, final dimension must be equal to ``embedding_dim``. All
                other leading dimensions will be flattened and treated as a large batch.
            is_training: boolean, whether this connection is to training data.

        Returns:
            dict: Dictionary containing the following keys and values:
                * ``quantize``: Tensor containing the quantized version of the input.
                * ``loss``: Tensor containing the loss to optimize.
                * ``encoding_indices``: Tensor containing the discrete encoding indices,
                ie which element of the quantized space each input element was mapped
                to.
        """
        # input shape A1 x ... x An x D
        # shape N x D, N = A1 * ... * An
        flattened = jnp.reshape(inputs, (-1, self.D))

        # shape N x 1
        flattened_sqr = jnp.sum(flattened**2, axis=-1, keepdims=True)

        # shape 1 x K
        codeboook_sqr = jnp.sum(self.codebook**2, axis=-1, keepdims=True).T

        # shape N x K
        # distances = (a-b)^2 = a^2 - 2*a*b + b^2
        distances = flattened_sqr - 2 * (flattened @ self.codebook.T) + codeboook_sqr

        # shape A1 x ... x An
        encoding_indices = jnp.reshape(
            jnp.argmin(distances, axis=-1), inputs.shape[:-1]
        )

        # shape A1 x ... x An x D
        quantize = self.codebook[encoding_indices]

        # loss = ||sg[z_e(x)] - e|| + beta||z_e(x) - sg[e]||
        encoding_loss = jnp.mean((jax.lax.stop_gradient(inputs) - quantize) ** 2)
        commit_loss = jnp.mean((inputs - jax.lax.stop_gradient(quantize)) ** 2)
        loss = encoding_loss + self.beta * commit_loss

        # straight-through estimator for reconstruction loss
        quantize = inputs + jax.lax.stop_gradient(quantize - inputs)

        return {
            "codebook_loss": loss,
            "quantize": quantize,
            "encoding_indices": encoding_indices,
        }

    def embed(self, indices):
        outshape = indices.shape + (self.D,)
        x = self.codebook[indices].reshape(outshape)
        return x
