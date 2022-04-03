from typing import Any
from pathlib import Path
import re

from tensorboardX import SummaryWriter

prefix = "exp"


def get_writer(base_dir: str, disable: bool = False) -> SummaryWriter:
    logdir = Path(base_dir)
    logdir.mkdir(parents=True, exist_ok=True)
    max_run = -1
    for path in logdir.iterdir():
        match = re.fullmatch(rf"{prefix}([0-9]+)", path.name)
        max_run = max(max_run, int(match.group(1)))
    return SummaryWriter(logdir=str(logdir / f"{prefix}{max_run+1}"),
                         flush_secs=10,
                         write_to_disk=not disable)


def log_dict(writer: SummaryWriter,
             logs: dict[str, Any],
             step: int,
             prefix: str = ""):
    for k, v in logs.items():
        assert isinstance(k, str)
        if k.startswith("scalar_"):
            k = k[len("scalar_"):]
            writer.add_scalar(f"{prefix}{k}", v, step)
        elif k.startswith("images_"):
            k = k[len("images_"):]
            writer.add_images(f"{prefix}{k}", v, step, dataformats="NHWC")
