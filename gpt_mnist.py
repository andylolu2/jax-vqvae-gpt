from pathlib import Path
import pickle
import json

import jax
import optax
from tqdm import tqdm

from dataset import load_vqvae_processed
from models.gpt import VqVaeGPTModel
from annotations import GPTConfig, VqVaeConfig
from logger import get_writer, log_dict


writer = get_writer("runs/gpt")
config = GPTConfig(
    seed=23,
    num_heads=4,
    hidden_dim=64,
    num_layers=3,
    dropout_rate=0.05,
    vqvae_config="runs/vqvae/exp0/config.json",
    train_dataset="datasets/exp0-encoded-train",
    test_dataset="datasets/exp0-encoded-test",
    train_steps=50000,
    test_steps=20,
    test_every=100,
    train_batch_size=64,
    test_batch_size=32,
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

optimizer = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
model = VqVaeGPTModel(label_classes,
                      vqvae_config,
                      config.num_heads,
                      config.hidden_dim,
                      config.num_layers,
                      config.dropout_rate,
                      optimizer)
train_state = model.initial_state(key, next(dset_train)[1])


for i in tqdm(range(config.train_steps)):
    epoch, batch = next(dset_train)
    train_state, logs = model.update(train_state, batch)

    writer.add_scalar("train/epoch", epoch, i)
    log_dict(writer, logs, step=i, prefix="train/")

    if (i + 1) % config.test_every == 0:
        for _ in range(config.test_steps):
            _, batch = next(dset_test)
            train_state, logs = model.evaluate(train_state, batch)
            log_dict(writer, logs, step=i, prefix="test/")

with open(Path(config.logdir) / config.output_name, "wb") as f:
    pickle.dump(train_state, f)

writer.close()
