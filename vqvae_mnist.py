from pathlib import Path
import pickle
import json
from collections import defaultdict

from tqdm import tqdm
import jax
import optax

from dataset import load_mnist
from models.vqvae import VqVaeModel
from logger import get_writer, log_dict
from annotations import VqVaeConfig


writer = get_writer("runs/vqvae")
config = VqVaeConfig(
    seed=23,
    K=64,
    D=256,
    compression_level=3,
    res_layers=1,
    commitment_loss=0.25,
    train_dset_percentage=100,
    test_dset_percentage=100,
    train_steps=10000,
    test_steps=20,
    test_every=200,
    train_batch_size=64,
    test_batch_size=64,
    learning_rate=3e-4,
    weight_decay=1e-5,
    logdir=writer.logdir,
    output_name="train_state.pkl"
)
with open(Path(config.logdir) / "config.json", "w") as f:
    json.dump(config._asdict(), f, indent=4)

key = jax.random.PRNGKey(config.seed)

_, dset_train = load_mnist(split="train",
                           batch_size=config.train_batch_size,
                           percentage=config.train_dset_percentage,
                           seed=config.seed)
_, dset_test = load_mnist(split="test",
                          batch_size=config.test_batch_size,
                          percentage=config.test_dset_percentage,
                          seed=config.seed)

optimizer = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
model = VqVaeModel(config.K,
                   config.D,
                   config.compression_level,
                   config.res_layers,
                   config.commitment_loss,
                   optimizer)
vqvae_state = model.initial_state(key, next(dset_train)[1])


for i in tqdm(range(config.train_steps)):
    epoch, batch = next(dset_train)
    vqvae_state, logs = model.update(vqvae_state, batch)

    writer.add_scalar("train/epoch", epoch, i)
    log_dict(writer, logs, step=i, prefix="train/")

    if (i + 1) % config.test_every == 0:
        logs = defaultdict(list)
        for _ in range(config.test_steps):
            _, batch = next(dset_test)
            log = model.evaluate(vqvae_state, batch)
            for k, v in log.items():
                logs[k].append(v)
        log_dict(writer, logs, step=i, prefix="test/")

with open(Path(config.logdir) / config.output_name, "wb") as f:
    pickle.dump(vqvae_state, f)

writer.close()
