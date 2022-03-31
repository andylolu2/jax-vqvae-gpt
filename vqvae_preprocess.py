from pathlib import Path
import pickle
import json

import datasets
from datasets.arrow_dataset import Dataset
import jax
import numpy as np

from models import build_model
from annotations import VqVaeConfig, TrainState, Batch

base_logdir = Path("runs")
exp = "exp13"
logdir = base_logdir / exp

with open(logdir / "config.json", "r") as f:
    config = VqVaeConfig(**json.load(f))
with open(logdir / config.output_name, "rb") as f:
    train_state: TrainState = pickle.load(f)

model = build_model(config.K, config.D)

dset = datasets.load_dataset(
    "mnist", split=f"train[:{config.train_dset_percentage}%]")


@jax.jit
def infer(train_state: TrainState, sample: Batch):
    params = train_state.params
    state = train_state.state
    result, _ = model.apply(params, state, sample, is_training=False)
    return result


def encode(batch):
    images = [np.array(img, dtype=np.float32) / 255 for img in batch["image"]]
    images = np.array(images)[..., None]
    result = infer(train_state, {"image": images})
    batch["encoding_indices"] = np.array(result["encoding_indices"])
    return batch


new_dset = dset.map(encode, batched=True, batch_size=64)
assert isinstance(new_dset, Dataset)
new_dset.save_to_disk(f"datasets/{exp}-encoded")
