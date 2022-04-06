import argparse
import json
from pathlib import Path
import pickle
import jax
from PIL import Image

import numpy as np

from annotations import GPTBatch, GPTConfig, GPTState, VqVaeConfig, VqVaeState
from trainers.gpt_trainer import VqVaeGPTTrainer
from trainers.vqvae_trainer import VqVaeTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MNIST samples by sampling VQ-VAE codes with a GPT-style transformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-p", "--path", type=str, required=True, default=argparse.SUPPRESS,
                        help="path to the pretrained GPT directory.")
    parser.add_argument("-o", "--out", type=str, required=True, default=argparse.SUPPRESS,
                        help="output directory to save generated images.")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="seed to sample results from.")
    parser.add_argument("-t", "--temperature", type=float, default=0.2,
                        help="temperature to sample results.")
    parser.add_argument("-S", "--samples", type=int, default=8,
                        help="Generates S x S samples.")
    return parser.parse_args()


def main(path: str, seed: int, temp: float, samples: int, out_path: str):
    model_dir = Path(path)

    with open(model_dir / "config.json", "r") as f:
        gpt_config = GPTConfig(**json.load(f))

    with open(gpt_config.vqvae_config, "r") as f:
        vqvae_config = VqVaeConfig(**json.load(f))
    with open(gpt_config.vqvae_state, "rb") as f:
        vqvae_state: VqVaeState = pickle.load(f)
    vqvae = VqVaeTrainer(K=vqvae_config.K,
                         D=vqvae_config.D,
                         compression_level=vqvae_config.compression_level,
                         res_layers=vqvae_config.res_layers,
                         commitment_loss=vqvae_config.commitment_loss,
                         optimizer=None)

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

    with open(model_dir / gpt_config.output_name, "rb") as f:
        gpt_state: GPTState = pickle.load(f)

    x = np.zeros((1, 32, 32, 1), dtype=np.float32)
    res, _ = vqvae.forward(vqvae_state.params, vqvae_state.state, x, False)
    sample: GPTBatch = {
        "encoding_indices": res["encoding_indices"],
        "label": np.zeros((1,))
    }
    gpt = VqVaeGPTTrainer(num_label_classes=10,
                          vqvae_config=vqvae_config,
                          num_heads=gpt_config.num_heads,
                          hidden_dim=gpt_config.hidden_dim,
                          num_layers=gpt_config.num_layers,
                          dropout_rate=gpt_config.dropout_rate,
                          sample=sample,
                          optimizer=None)
    rng = jax.random.PRNGKey(seed)
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label in range(10):
        image = np.zeros((samples * 32, samples * 32), dtype=np.uint8)
        for i in range(samples):
            for j in range(samples):
                indices, rng = gpt.generate(gpt_state, rng, label, temp=temp)
                img = decode_indices(vqvae_state, indices)
                img = (img[0, :, :, 0] * 255).astype(np.uint8)
                x, y = i * 32, j * 32
                image[x:x + 32, y:y + 32] = img
        im = Image.fromarray(image)
        im.save(str(out_dir / f"generated_{label}.png"))


if __name__ == "__main__":
    args = parse_args()
    main(path=args.path,
         out_path=args.out,
         seed=args.seed,
         temp=args.temperature,
         samples=args.samples)