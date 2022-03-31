from typing import Optional

import haiku as hk
import jax
import jax.nn as nn
import jax.numpy as jnp

from annotations import Batch


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
    def __init__(self, out_channels: int, kernel_size: int = 3, name: Optional[str] = None):
        super().__init__(name)
        self.conv1 = hk.Conv2D(out_channels // 2, kernel_size, stride=2)
        self.batchnorm1 = hk.BatchNorm(True, True, 0.9)
        self.conv2 = hk.Conv2D(out_channels, kernel_size, stride=2)
        self.batchnorm2 = hk.BatchNorm(True, True, 0.9)
        self.res = ResBlock(out_channels, kernel_size)

    def __call__(self, x, is_training: bool) -> jnp.ndarray:
        x = self.conv1(x)
        x = self.batchnorm1(x, is_training)
        x = nn.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x, is_training)
        x = nn.relu(x)
        x = self.res(x, is_training)
        return x


class CnnDencoder(hk.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, name: Optional[str] = None):
        super().__init__(name)
        self.res = ResBlock(in_channels, kernel_size)
        self.deconv1 = hk.Conv2DTranspose(
            in_channels // 2, kernel_size, stride=2)
        self.batchnorm1 = hk.BatchNorm(True, True, 0.9)
        self.deconv2 = hk.Conv2DTranspose(1, kernel_size, stride=2)
        self.batchnorm2 = hk.BatchNorm(True, True, 0.9)

    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        x = self.res(x, is_training)
        x = self.deconv1(x)
        x = self.batchnorm1(x, is_training)
        x = nn.relu(x)
        x = self.deconv2(x)
        x = self.batchnorm2(x, is_training)
        x = nn.sigmoid(x)
        return x


class QuantizedCodebook(hk.Module):
    def __init__(self,
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

    def __call__(self, inputs):
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


class VqVae(hk.Module):
    def __init__(self,
                 embed_size_K: int,
                 embed_dim_D: int,
                 commitment_loss: float = 0.25,
                 name: Optional[str] = None):
        super().__init__(name)
        self.K = embed_size_K
        self.D = embed_dim_D
        self.encoder = CnnEncoder(self.D)
        self.decoder = CnnDencoder(self.D)

        self.codebook = QuantizedCodebook(self.K, self.D, commitment_loss)

    def __call__(self, x, is_training: bool):
        encodings = self.encoder(x, is_training)

        result = self.codebook(encodings)
        reconstruction = self.decoder(result["quantize"], is_training)

        result["reconstruction"] = reconstruction

        return result


def build_model(K: int, D: int):
    @hk.without_apply_rng
    @hk.transform_with_state
    def model(batch: Batch, is_training: bool):
        x = batch["image"]
        net = VqVae(K, D)
        return net(x, is_training)
    return model
