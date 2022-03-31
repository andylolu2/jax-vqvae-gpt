from pathlib import Path
import re

from tensorboardX import SummaryWriter

logdir = Path("runs")
prefix = "exp"


def get_writer() -> SummaryWriter:
    logdir.mkdir(parents=True, exist_ok=True)
    max_run = -1
    for path in logdir.iterdir():
        match = re.fullmatch(rf"{prefix}([0-9]+)", path.name)
        max_run = max(max_run, int(match.group(1)))
    return SummaryWriter(logdir=str(logdir / f"{prefix}{max_run+1}"),
                         flush_secs=10)
