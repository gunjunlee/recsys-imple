import random
import logging
from datetime import datetime

import numpy as np

import torch

log = logging.getLogger(__name__)


def fix_rng(random_seed):
    # Plain python
    random.seed(random_seed)

    # Numpy
    np.random.seed(random_seed)

    # Torch
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Timer():
    def __init__(self, desc=""):
        self.desc = desc

    def __enter__(self):
        self.start_dt = datetime.now()
        log.info(f"{self.desc} start")
        return self

    def __exit__(self, type, value, traceback):
        end_dt = datetime.now()
        elapsed = end_dt - self.start_dt
        log.info(f"{self.desc} done (elapsed time: {elapsed.total_seconds():.2f}s)")
