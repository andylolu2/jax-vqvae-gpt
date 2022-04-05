from pathlib import Path
import pickle
import json
from collections import defaultdict

import jax
import optax
from tqdm import tqdm

from dataset import load_vqvae_processed
from models.gpt import VqVaeGPTModel
from models.vqvae import VqVaeModel
from annotations import GPTConfig, VqVaeConfig, VqVaeState
from logger import get_writer, log_dict


writer = get_writer("runs/gpt")
config = GPTConfig(
    seed=23,
    num_heads=4,
    hidden_dim=64,
    num_layers=4,
    dropout_rate=0.1,
    vqvae_config="runs/vqvae/exp1/config.json",
    vqvae_state="runs/vqvae/exp1/train_state.pkl",
    train_dataset="datasets/exp1-encoded-train",
    test_dataset="datasets/exp1-encoded-test",
    train_steps=5000,
    test_steps=20,
    test_every=500,
    train_batch_size=64,
    test_batch_size=64,
    learning_rate=3e-4,
    weight_decay=1e-5,
    logdir=writer.logdir,
    output_name="train_state.pkl"
)
with open(Path(config.logdir) / "config.json", "w") as f:
    json.dump(config._asdict(), f, indent=4)
with open(config.vqvae_config, "r") as f:
    vqvae_config = VqVaeConfig(**json.load(f))

key = jax.random.PRNGKey(config.seed)

features, dset_train = load_vqvae_processed(
    path=config.train_dataset,
    batch_size=config.train_batch_size,
    repeat=True,
    seed=config.seed
)

_, dset_test = load_vqvae_processed(
    path=config.test_dataset,
    batch_size=config.test_batch_size,
    repeat=True,
    seed=config.seed
)

label_classes = features["label"].num_classes

_, sample = next(dset_train)
optimizer = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
model = VqVaeGPTModel(label_classes,
                      vqvae_config,
                      config.num_heads,
                      config.hidden_dim,
                      config.num_layers,
                      config.dropout_rate,
                      sample,
                      optimizer)
train_state = model.initial_state(key, sample)


vqvae = VqVaeModel(vqvae_config.K,
                   vqvae_config.D,
                   vqvae_config.compression_level,
                   vqvae_config.res_layers,
                   vqvae_config.commitment_loss,
                   None)
with open(config.vqvae_state, "rb") as f:
    vqvae_state: VqVaeState = pickle.load(f)


@jax.jit
def decode_indices(vqvae_state: VqVaeState, indices):
    z_q = vqvae.lookup_indices(vqvae_state, indices)
    img, _ = vqvae.apply.decode(
        vqvae_state.params,
        vqvae_state.state,
        None,
        z_q,
        is_training=False)
    return img


for i in tqdm(range(config.train_steps)):
    epoch, batch = next(dset_train)
    train_state, logs = model.update(train_state, batch)

    writer.add_scalar("train/epoch", epoch, i)
    log_dict(writer, logs, step=i, prefix="train/")

    if (i + 1) % config.test_every == 0:
        logs = defaultdict(list)
        for _ in range(config.test_steps):
            _, batch = next(dset_test)
            train_state, log = model.evaluate(train_state, batch)
            for k, v in log.items():
                logs[k].append(v)
        log_dict(writer, logs, step=i, prefix="test/")
        for label in range(label_classes):
            indices = model.generate(train_state, label)
            img = decode_indices(vqvae_state, indices)
            writer.add_image(
                f"test/generate_{label}", img[0], i, dataformats="HWC"
            )

with open(Path(config.logdir) / config.output_name, "wb") as f:
    pickle.dump(train_state, f)

writer.close()
