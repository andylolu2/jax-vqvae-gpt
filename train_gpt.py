from typing import Optional
from pathlib import Path
import pickle
import json
import argparse
from collections import defaultdict

import jax
import optax
from tqdm import tqdm
import numpy as np

from trainers.gpt_trainer import VqVaeGPTTrainer
from trainers.vqvae_trainer import VqVaeTrainer
from utils.dataset import load_vqvae_processed
from utils.annotations import GPTConfig, GPTState, VqVaeConfig, VqVaeState
from utils.logger import get_writer, log_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a GPT-style transformer the VQ-VAE tokens of the MNIST dataset."
    )
    parser.add_argument(
        "-f", "--file", type=str, required=True, help="path to the json config file."
    )
    parser.add_argument(
        "-chkp", "--checkpoint", type=str, help="path to train state pkl file."
    )
    return parser.parse_args()


def main(config: GPTConfig, checkpoint: Optional[str] = None):
    writer = get_writer(config.logdir)
    exp_dir = writer.logdir

    # save config file
    with open(Path(exp_dir) / "config.json", "w") as f:
        json.dump(config._asdict(), f, indent=4)

    # load dataset
    features, dset_train = load_vqvae_processed(
        path=config.train_dataset,
        batch_size=config.train_batch_size,
        repeat=True,
        seed=config.seed,
    )
    _, dset_test = load_vqvae_processed(
        path=config.test_dataset,
        batch_size=config.test_batch_size,
        repeat=True,
        seed=config.seed,
    )
    label_classes = features["label"].num_classes

    # load vqvae for evaluation
    with open(config.vqvae_config, "r") as f:
        vqvae_config = VqVaeConfig(**json.load(f))
    with open(config.vqvae_state, "rb") as f:
        vqvae_state: VqVaeState = pickle.load(f)
    vqvae = VqVaeTrainer(
        K=vqvae_config.K,
        D=vqvae_config.D,
        compression_level=vqvae_config.compression_level,
        res_layers=vqvae_config.res_layers,
        commitment_loss=vqvae_config.commitment_loss,
        optimizer=None,
    )

    @jax.jit
    def decode_indices(vqvae_state: VqVaeState, indices):
        z_q = vqvae.lookup_indices(vqvae_state, indices)
        img, _ = vqvae.apply.decode(
            vqvae_state.params, vqvae_state.state, None, z_q, is_training=False
        )
        return img

    # initialize model
    _, sample = next(dset_train)
    optimizer = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    trainer = VqVaeGPTTrainer(
        label_classes,
        vqvae_config,
        config.num_heads,
        config.hidden_dim,
        config.num_layers,
        config.dropout_rate,
        sample,
        optimizer,
    )
    key = jax.random.PRNGKey(config.seed)
    if checkpoint is None:
        key, key1 = jax.random.split(key)
        train_state = trainer.initial_state(key1, sample)
    else:
        with open(checkpoint, "rb") as f:
            train_state: GPTState = pickle.load(f)

    # training loop
    for i in tqdm(range(config.train_steps)):
        # update
        epoch, batch = next(dset_train)
        train_state, logs = trainer.update(train_state, batch)

        # log
        logs["scalar_epoch"] = epoch
        log_dict(writer, logs, step=i, prefix="train/")

        # evaluate
        if (i + 1) % config.test_every == 0:
            # test loss
            logs = defaultdict(list)
            for _ in range(config.test_steps):
                _, batch = next(dset_test)
                train_state, log = trainer.evaluate(train_state, batch)
                for k, v in log.items():
                    logs[k].append(v)
            log_dict(writer, logs, step=i, prefix="test/")

            # generate samples
            for label in range(label_classes):
                images = []
                for _ in range(config.generate_samples):
                    indices, key = trainer.generate(
                        train_state, key, label, temp=config.sample_temperature
                    )
                    img = decode_indices(vqvae_state, indices)
                    images.append(img[0])
                images = np.array(images)
                logs = {f"images_generate_{label}": images}
                log_dict(writer, logs, step=i, prefix="test/")

    with open(Path(exp_dir) / config.output_name, "wb") as f:
        pickle.dump(train_state, f)

    writer.close()


if __name__ == "__main__":
    args = parse_args()
    with open(args.file, "r") as f:
        config = GPTConfig(**json.load(f))

    main(config, checkpoint=args.checkpoint)
