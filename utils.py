from munch import Munch, munchify

import random
import numpy as np
import torch
import yaml

from data_loader import DataLoader
from model import GPT, ModelArgs
from tokenizer import Tokenizer
from trainer import TrainArgs, Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_global_config(path: str):
    with open(path, 'rb') as f:
        config = yaml.safe_load(f)
    config = munchify(config)
    return config


def init_by_config_path(input_path: str, config_path: str):
    config = get_global_config(config_path)
    set_seed(config.seed)

    tokenizer = Tokenizer(config.vocab_file)
    config.model.vocab_size = tokenizer.vocab_size

    config.bos_idx = tokenizer.encode(tokenizer.bos_token)
    config.eos_idx = tokenizer.encode(tokenizer.eos_token)

    data_loader = DataLoader(
        input_path, tokenizer, config.model.block_size
    )

    model_args = ModelArgs.from_dict(config.model)
    model_args.max_batch_size = config.model.max_batch_size
    model = GPT(model_args)

    trainer_args = TrainArgs.from_dict(config.trainer)
    trainer = Trainer(trainer_args, model)

    return config, tokenizer, data_loader, trainer
