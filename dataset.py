from typing import Iterator, Optional
from itertools import count

import numpy as np
import datasets
from datasets.arrow_dataset import Dataset

from annotations import Batch


def preprocess(batch) -> Batch:
    images = [np.array(img, dtype=np.float32) / 255 for img in batch["image"]]
    images = np.array(images)[..., None]
    return {
        "image": images,
        "label": np.array(batch["label"])
    }


def load_mnist(split: str,
               batch_size: int,
               percentage: int,
               repeat: bool = True,
               seed: Optional[int] = None) -> Iterator[tuple[int, Batch]]:
    dset = datasets.load_dataset("mnist", split=f"{split}[:{percentage}%]")
    assert isinstance(dset, Dataset)

    dset.set_transform(preprocess)

    counter = count()
    while True:
        dset = dset.shuffle(seed)
        epoch = next(counter)
        for i in range(0, len(dset) - batch_size, batch_size):
            yield epoch, dset[i: i + batch_size]

        if not repeat:
            break
