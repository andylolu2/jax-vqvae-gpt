from pathlib import Path
import pickle
import json

from tqdm import tqdm
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from dataset import load_mnist, Batch
from models.vqvae import build_model
from logger import get_writer
from annotations import VqVaeConfig, TrainState, Batch


writer = get_writer()
config = VqVaeConfig(
    seed=23,
    K=128,
    D=64,
    train_dset_percentage=100,
    test_dset_percentage=100,
    train_steps=10000,
    test_steps=1,
    log_every=200,
    train_batch_size=32,
    test_batch_size=32,
    learning_rate=3e-4,
    weight_decay=1e-5,
    logdir=writer.logdir,
    output_name="train_state.pkl"
)
with open(Path(config.logdir) / "config.json", "w") as f:
    json.dump(config._asdict(), f, indent=4)

key = jax.random.PRNGKey(config.seed)

model = build_model(config.K, config.D)


def mse(x, x_pred):
    return jnp.mean((x - x_pred) ** 2)


def loss(params: hk.Params, state: hk.State, batch: Batch, is_training: bool):
    result, new_state = model.apply(params, state, batch, is_training)
    reconstruction_loss = mse(batch["image"], result["reconstruction"])
    loss = reconstruction_loss + result["codebook_loss"]
    return loss, (new_state, result)


loss_and_grad = jax.value_and_grad(loss, has_aux=True)
optimizer = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)

mnist_train = load_mnist("train",
                         config.train_batch_size,
                         percentage=config.train_dset_percentage,
                         seed=config.seed)
mnist_test = load_mnist("test",
                        config.test_batch_size,
                        percentage=config.test_dset_percentage,
                        seed=config.seed)


@jax.jit
def step(train_state: TrainState, batch: Batch) -> tuple[TrainState, jnp.ndarray, dict]:
    (loss_val, (state, result)), grads = loss_and_grad(train_state.params,
                                                       train_state.state,
                                                       batch,
                                                       True)
    updates, opt_state = optimizer.update(grads,
                                          train_state.opt_state,
                                          train_state.params)
    params = optax.apply_updates(train_state.params, updates)
    train_state = TrainState(params, state, opt_state)
    return train_state, loss_val, result


@jax.jit
def eval_step(train_state: TrainState, batch: Batch) -> tuple[TrainState, jnp.ndarray, dict]:
    loss_val, (state, result) = loss(train_state.params,
                                     train_state.state,
                                     batch,
                                     False)
    train_state = TrainState(train_state.params, state, train_state.opt_state)
    return train_state, loss_val, result


key, key1 = jax.random.split(key)
_, sample_batch = next(mnist_train)
params, state = model.init(key1, sample_batch, is_training=True)
opt_state = optimizer.init(params)

train_state = TrainState(params, state, opt_state)
print("Initialized model!")


for i in tqdm(range(config.train_steps)):
    epoch, batch = next(mnist_train)
    train_state, loss_val, _ = step(train_state, batch)

    writer.add_scalar("train/loss", jax.device_get(loss_val), i)
    writer.add_scalar("train/epoch", epoch, i)

    if i % config.log_every == 0:
        for _ in range(config.test_steps):
            _, batch = next(mnist_test)
            train_state, loss_val, result = eval_step(train_state, batch)

            writer.add_scalar("test/loss", jax.device_get(loss_val), i)
            writer.add_images(
                "test/original", batch["image"], i, dataformats="NHWC")
            writer.add_images("test/reconstruction",
                              result["reconstruction"], i, dataformats="NHWC")

with open(Path(config.logdir) / config.output_name, "wb") as f:
    pickle.dump(train_state, f)

writer.close()
