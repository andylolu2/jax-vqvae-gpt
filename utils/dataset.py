from typing import Iterator, Optional, Union
from itertools import count
from pathlib import Path
from datasets.features.features import Features

import numpy as np
import datasets
from datasets.arrow_dataset import Dataset
from skimage.transform import resize

from utils.annotations import VqVaeBatch, GPTBatch


def process_image(img, shape: tuple[int, int]) -> np.ndarray:
    img = np.array(img, dtype=np.float32) / 255
    img = resize(img, shape)
    img = img[..., None]
    return img


def load_dset(
    name: str,
    split: str,
    batch_size: int,
    percentage: int,
    resize_shape: tuple[int, int],
    repeat: bool = True,
    seed: Optional[int] = None,
) -> tuple[Features, Iterator[tuple[int, VqVaeBatch]]]:
    """
    Loads a dataset with preprocessing, batching, and repeating.

    Args:
        name (str): The name of the dataset on Hugging Face Hub.
        split (str): The split of the dataset, such as "train" / "test".
        batch_size (int): The batch size to load the data.
        percentage (int): The percentage of the dataset to use.
        resize_shape (tuple[int, int], optional): Shape to resize the image to.
        repeat (bool, optional): Whether or not to repeat the dataset after
            iterating through one epoch. Defaults to True.
        seed (Optional[int], optional): The seed used to suffle the dataset. Defaults to None.

    Returns:
        tuple[Features, Iterator[tuple[int, VqVaeBatch]]]: A tuple of dataset features and the
            iterator which yields preprocessed, batched, and repeated data.
    """
    dset = datasets.load_dataset(name, split=f"{split}[:{percentage}%]")
    assert isinstance(dset, Dataset)

    features: Features = dset.features

    def preprocess(batch) -> VqVaeBatch:
        return {
            "image": np.array(
                [process_image(img, resize_shape) for img in batch["image"]]
            ),
            "label": np.array(batch["label"]),
        }

    dset.set_transform(preprocess)

    def iterator(dset: Dataset) -> Iterator[tuple[int, VqVaeBatch]]:
        counter = count()
        while True:
            dset = dset.shuffle(seed)
            epoch = next(counter)
            for i in range(0, len(dset) - batch_size, batch_size):
                yield epoch, dset[i : i + batch_size]

            if not repeat:
                break

    return features, iterator(dset)


def load_vqvae_processed(
    path: Union[str, Path],
    batch_size: int,
    repeat: bool = True,
    seed: Optional[int] = None,
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
                yield epoch, dset[i : i + batch_size]

            if not repeat:
                break

    return features, iterator(dset)
