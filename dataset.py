from typing import Iterator, Optional, Union
from itertools import count
from pathlib import Path
from datasets.features.features import Features

import numpy as np
import datasets
from datasets.arrow_dataset import Dataset
from skimage.transform import resize

from annotations import VqVaeBatch, GPTBatch


def load_mnist(split: str,
               batch_size: int,
               percentage: int,
               repeat: bool = True,
               seed: Optional[int] = None) -> tuple[Features, Iterator[tuple[int, VqVaeBatch]]]:
    dset = datasets.load_dataset("mnist", split=f"{split}[:{percentage}%]")
    assert isinstance(dset, Dataset)

    features: Features = dset.features

    def preprocess(batch) -> VqVaeBatch:
        images = []
        for img in batch["image"]:
            arr = np.array(img, dtype=np.float32) / 255
            images.append(resize(arr, (32, 32)))
        images = np.array(images)[..., None]
        return {
            "image": images,
            "label": np.array(batch["label"])
        }
    dset.set_transform(preprocess)

    def iterator(dset: Dataset) -> Iterator[tuple[int, VqVaeBatch]]:
        counter = count()
        while True:
            dset = dset.shuffle(seed)
            epoch = next(counter)
            for i in range(0, len(dset) - batch_size, batch_size):
                yield epoch, dset[i: i + batch_size]

            if not repeat:
                break
    return features, iterator(dset)


def load_vqvae_processed(path: Union[str, Path],
                         batch_size: int,
                         repeat: bool = True,
                         seed: Optional[int] = None
                         ) -> tuple[Features, Iterator[tuple[int, GPTBatch]]]:
    dset = datasets.load.load_from_disk(str(path))
    assert isinstance(dset, Dataset)

    features: Features = dset.features

    def preprocess(batch) -> GPTBatch:
        return {
            "encoding_indices": np.array(batch["encoding_indices"]),
            "label": np.array(batch["label"]),
        }
    dset.set_transform(preprocess)

    def iterator(dset: Dataset) -> Iterator[tuple[int, GPTBatch]]:
        counter = count()
        while True:
            dset = dset.shuffle(seed)
            epoch = next(counter)
            for i in range(0, len(dset) - batch_size, batch_size):
                yield epoch, dset[i: i + batch_size]

            if not repeat:
                break

    return features, iterator(dset)
