from pathlib import Path
import pickle
import json
from PIL import Image

import numpy as np

from models.vqvae import build_model
from dataset import load_mnist
from annotations import VqVaeConfig, VqVaeState, VqVaeBatch

logdir = Path("runs/exp13")
with open(logdir / "config.json", "r") as f:
    config = VqVaeConfig(**json.load(f))
with open(logdir / config.output_name, "rb") as f:
    train_state: VqVaeState = pickle.load(f)

model = build_model(config.K, config.D)

mnist_train = load_mnist("train", 1, 100, seed=0)
_, sample = next(mnist_train)


def infer(train_state: VqVaeState, sample: VqVaeBatch):
    params = train_state.params
    state = train_state.state
    result, _ = model.apply(params, state, sample, is_training=False)
    return result


x1 = sample["image"][0, ..., 0] * 255

x2 = infer(train_state, sample)["reconstruction"]
x2 = np.array(x2)[0, ..., 0] * 255

combined = np.concatenate((x1, x2), axis=-1)

im = Image.fromarray(combined)
im.show()
