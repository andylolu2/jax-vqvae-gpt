import haiku as hk
import jax
import jax.nn as nn
import jax.numpy as jnp
import optax

from dataset import load_mnist
from models import MySequential

seed = 23

jax.config.update('jax_platform_name', 'cpu')
key = jax.random.PRNGKey(seed)


num_classes = 10


@hk.without_apply_rng
@hk.transform
def model(batch):
    x = batch["image"]
    net = MySequential([
        hk.Conv2D(64, [3, 3], 2),
        nn.relu,
        hk.Conv2D(64, [3, 3], 2),
        nn.relu,
        hk.Flatten(),
        hk.Linear(num_classes),
    ])
    return net(x)


optimizer = optax.adamw(3e-4)


def cross_entropy(y_true, y_pred):
    return jnp.sum(-y_true * nn.log_softmax(y_pred), axis=-1)


@jax.value_and_grad
def loss(params: hk.Params, batch):
    y = nn.one_hot(batch["label"], num_classes)
    y_pred = model.apply(params, batch)
    loss = jnp.mean(cross_entropy(y, y_pred), axis=0)
    return loss


@jax.jit
def step(params: hk.Params, opt_state: optax.OptState, batch):
    loss_value, grads = loss(params, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_value


@jax.jit
def accuracy(params: hk.Params, batch):
    pred = jnp.argmax(model.apply(params, batch), axis=-1)
    return jnp.mean(pred == batch["label"])


train_steps = 10000
test_steps = 100

log_every = 100
batch_size = 32

mnist_train = load_mnist("train", batch_size, percentage=10)
mnist_test = load_mnist("test", batch_size, percentage=10)

key, key1 = jax.random.split(key)
params = model.init(key1, next(mnist_train))
opt_state = optimizer.init(params)

losses = []

for i in range(train_steps):
    params, opt_state, loss_val = step(params, opt_state, next(mnist_train))

    losses.append(jax.device_get(loss_val))
    if i % log_every == 0:
        accs = []
        for _ in range(test_steps):
            acc = accuracy(params, next(mnist_test))
            accs.append(jax.device_get(acc))

        acc = sum(accs) / len(accs)
        print(f"Test accuracy: {acc:.3f}")

        loss_val = sum(losses) / len(losses)
        print(f"Test loss: {loss_val:.3f}")
        losses = []
