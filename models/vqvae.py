from typing import Optional

import haiku as hk
import jax
import jax.nn as nn
import jax.numpy as jnp


class ResBlock(hk.Module):
    def __init__(self, dim: int, kernel_size: int, name: Optional[str] = None):
        super().__init__(name)
        self.conv1 = hk.Conv2D(dim, kernel_size, stride=1)
        self.batchnorm1 = hk.BatchNorm(True, True, 0.9)
        self.conv2 = hk.Conv2D(dim, kernel_size, stride=1)
        self.batchnorm2 = hk.BatchNorm(True, True, 0.9)

    def __call__(self, x, is_training: bool) -> jnp.ndarray:
        res = self.conv1(x)
        res = self.batchnorm1(res, is_training)
        res = nn.relu(res)
        res = self.conv2(res)
        res = self.batchnorm2(res, is_training)
        x += res
        x = nn.relu(x)
        return x


class CnnEncoder(hk.Module):
    def __init__(
            self,
            out_channels: int,
            downscale_level: int,
            res_layers: int = 1,
            kernel_size: int = 3,
            name: Optional[str] = None):
        super().__init__(name)
        self.downscale_layers = []
        for i in range(downscale_level - 1, -1, -1):
            num_channels = out_channels // (2**i)
            self.downscale_layers.append((
                hk.Conv2D(num_channels, kernel_size, stride=2),
                hk.BatchNorm(True, True, 0.9)
            ))
        self.res_layers = [
            ResBlock(out_channels, kernel_size) for _ in range(res_layers)
        ]

    def __call__(self, x, is_training: bool) -> jnp.ndarray:
        for conv, bn in self.downscale_layers:
            x = conv(x)
            x = bn(x, is_training)
            x = nn.relu(x)
        for res in self.res_layers:
            x = res(x, is_training)
        return x


class CnnDecoder(hk.Module):
    def __init__(
            self,
            in_channels: int,
            upscale_level: int,
            res_layers: int = 1,
            kernel_size: int = 3,
            name: Optional[str] = None):
        super().__init__(name)
        self.res_layers = [
            ResBlock(in_channels, kernel_size) for _ in range(res_layers)
        ]
        self.upscale_layers = []
        for i in range(upscale_level - 1):
            num_channels = in_channels // (2**i)
            self.upscale_layers.append((
                hk.Conv2DTranspose(num_channels, kernel_size, stride=2),
                hk.BatchNorm(True, True, 0.9)
            ))
        self.out = hk.Conv2DTranspose(1, kernel_size, stride=2)
        self.bn_out = hk.BatchNorm(True, True, 0.9)

    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        for res in self.res_layers:
            x = res(x, is_training)
        for deconv, bn in self.upscale_layers:
            x = deconv(x)
            x = bn(x, is_training)
            x = nn.relu(x)
        x = self.out(x)
        x = self.bn_out(x, is_training)
        x = nn.sigmoid(x)
        return x


class QuantizedCodebook(hk.Module):
    def __init__(
            self,
            embed_size_K: int,
            embed_dim_D: int,
            commitment_loss: float,
            name: Optional[str] = None):
        super().__init__(name)
        self.K = embed_size_K
        self.D = embed_dim_D
        self.beta = commitment_loss

        initializer = hk.initializers.VarianceScaling(distribution="uniform")
        self.codebook = hk.get_parameter(
            "codebook", (self.K, self.D), init=initializer)

    def __call__(self, inputs) -> dict[str, jnp.ndarray]:
        '''Connects the module to some inputs.

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
        '''
        # input shape A1 x ... x An x D
        # shape N x D, N = A1 * ... * An
        flattened = jnp.reshape(inputs, (-1, self.D))

        # shape N x 1
        flattened_sqr = jnp.sum(flattened ** 2, axis=-1, keepdims=True)

        # shape 1 x K
        codeboook_sqr = jnp.sum(self.codebook ** 2, axis=-1, keepdims=True).T

        # shape N x K
        # distances = (a-b)^2 = a^2 - 2*a*b + b^2
        distances = flattened_sqr - \
            2 * (flattened @ self.codebook.T) + codeboook_sqr

        # shape A1 x ... x An
        encoding_indices = jnp.reshape(jnp.argmin(
            distances, axis=-1), inputs.shape[:-1])

        # shape A1 x ... x An x D
        quantize = self.codebook[encoding_indices]
        # straight-through estimator
        quantize = inputs + jax.lax.stop_gradient(quantize - inputs)

        # loss = ||sg[z_e(x)] - e|| + beta||z_e(x) - sg[e]||
        encoding_loss = jnp.mean((jax.lax.stop_gradient(quantize) - inputs)**2)
        commit_loss = jnp.mean((quantize - jax.lax.stop_gradient(inputs))**2)
        loss = encoding_loss + self.beta * commit_loss

        return {
            "codebook_loss": loss,
            "quantize": quantize,
            "encoding_indices": encoding_indices,
        }

    def embed(self, indices):
        outshape = indices.shape + (self.D,)
        x = self.codebook[indices].reshape(outshape)
        return x
