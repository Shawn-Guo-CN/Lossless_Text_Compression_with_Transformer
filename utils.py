import random
import numpy as np
import tomllib
import torch
from munch import Munch, munchify


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_global_config(path: str):
    with open(path, 'rb') as f:
        config = tomllib.load(f)
    config = munchify(config)
    assert config.max_input_length == config.model.block_size, \
        "max_input_length should be equal to model.block_size"
    return 
