import pickle
import json
import argparse
from pathlib import Path

import datasets
import jax
import numpy as np

from trainers.vqvae_trainer import VqVaeTrainer
from utils.dataset import process_image
from utils.annotations import VqVaeConfig, VqVaeState


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode the MNIST dataset with a VQ-VAE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        default=argparse.SUPPRESS,
        help="path to directory of the trained VQ-VAE model.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        required=True,
        default=argparse.SUPPRESS,
        help="path to directory to save the processed datasets.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=64,
        help="batch size to process the dataset with.",
    )
    parser.add_argument(
        "-P",
        "--percentage",
        type=int,
        default=100,
        help="percentage of dataset to encode.",
    )
    return parser.parse_args()


def main(path: str, out_path: str, batch_size: int, percentage: int):
    model_dir = Path(path)

    with open(model_dir / "config.json", "r") as f:
        config = VqVaeConfig(**json.load(f))
    with open(model_dir / config.output_name, "rb") as f:
        vqvae_state: VqVaeState = pickle.load(f)

    trainer = VqVaeTrainer(
        K=config.K,
        D=config.D,
        compression_level=config.compression_level,
        res_layers=config.res_layers,
        commitment_loss=config.commitment_loss,
        optimizer=None,
    )

    @jax.jit
    def infer(vqvae_state: VqVaeState, x: np.ndarray):
        params, state = vqvae_state.params, vqvae_state.state
        z_e, _ = trainer.apply.encode(params, state, None, x, is_training=False)
        result, _ = trainer.apply.quantize(params, state, None, z_e)
        indices = result["encoding_indices"]

        # z1, z2 are not necessary but used for assertion
        z1 = trainer.lookup_indices(vqvae_state, indices)
        z2 = result["quantize"]

        return result, (z1, z2)

    def encode(batch):
        images = np.array(
            [process_image(img, shape=config.resize_shape) for img in batch["image"]]
        )
        result, (z1, z2) = infer(vqvae_state, images)
        batch["encoding_indices"] = np.array(result["encoding_indices"])

        assert np.allclose(z1, z2, atol=1e-6, rtol=0)
        assert batch["encoding_indices"].ndim == 3
        assert batch["encoding_indices"].dtype == np.int32
        assert np.max(batch["encoding_indices"]) < config.K
        assert np.min(batch["encoding_indices"]) >= 0

        return batch

    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        dset = datasets.load_dataset("mnist", split=f"{split}[:{percentage}%]")
        dset = dset.map(encode, batched=True, batch_size=batch_size)
        dset.save_to_disk(str(out_dir / split))


if __name__ == "__main__":
    args = parse_args()
    main(
        path=args.path,
        out_path=args.out,
        batch_size=args.batch_size,
        percentage=args.percentage,
    )
