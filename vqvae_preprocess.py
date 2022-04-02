from pathlib import Path
import pickle
import json

import datasets
from datasets.arrow_dataset import Dataset
import jax
import jax.numpy as jnp
import numpy as np

from models.vqvae import VqVaeModel
from annotations import VqVaeConfig, VqVaeState

base_logdir = Path("runs/vqvae")
exp = "exp0"

logdir = base_logdir / exp

with open(logdir / "config.json", "r") as f:
    config = VqVaeConfig(**json.load(f))
with open(logdir / config.output_name, "rb") as f:
    vqvae_state: VqVaeState = pickle.load(f)

model = VqVaeModel(config.K, config.D, config.commitment_loss, None)


@jax.jit
def infer(vqvae_state: VqVaeState, batch):
    params, state = vqvae_state.params, vqvae_state.state
    x = batch["image"]
    z_e, _ = model.apply.encode(params, state, None, x, is_training=False)
    result, _ = model.apply.quantize(params, state, None, z_e)

    indices = result["encoding_indices"]
    z1 = model.lookup_indices(vqvae_state, indices)
    z2 = result["quantize"]
    return result, (z1, z2)


def encode(batch):
    images = [np.array(img, dtype=np.float32) / 255 for img in batch["image"]]
    images = np.array(images)[..., None]
    result, (z1, z2) = infer(vqvae_state, {"image": images})
    batch["encoding_indices"] = np.array(result["encoding_indices"])

    assert np.allclose(z1, z2, atol=1e-6, rtol=0)
    assert batch["encoding_indices"].ndim == 3
    assert batch["encoding_indices"].dtype == np.int32
    assert np.max(batch["encoding_indices"]) < config.K
    assert np.min(batch["encoding_indices"]) >= 0

    return batch


if __name__ == "__main__":
    dset = datasets.load_dataset(
        "mnist", split=f"train[:{config.train_dset_percentage}%]")
    new_dset = dset.map(encode, batched=True,
                        batch_size=config.train_batch_size)
    assert isinstance(new_dset, Dataset)
    new_dset.save_to_disk(f"datasets/{exp}-encoded-train")

    dset = datasets.load_dataset(
        "mnist", split=f"test[:{config.test_dset_percentage}%]")
    new_dset = dset.map(encode, batched=True,
                        batch_size=config.train_batch_size)
    assert isinstance(new_dset, Dataset)
    new_dset.save_to_disk(f"datasets/{exp}-encoded-test")
