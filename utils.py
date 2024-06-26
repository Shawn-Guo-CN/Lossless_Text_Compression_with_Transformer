import argparse
from munch import munchify
import numpy as np
import random
import torch
from tqdm import tqdm
import yaml

import data_loader
import file_io
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


def init_by_config_path(
    input_path: str,
    output_path: str,
    config_path: str,
    mode: str,
):
    assert mode == 'compress' or mode == 'decompress', \
        f'Invalid mode {mode}. Choose either \'compress\' or \'decompress\'.'
    config = get_global_config(config_path)
    set_seed(config.seed)

    tokenizer = Tokenizer(config.vocab_file)
    config.model.vocab_size = tokenizer.vocab_size

    config.bos_idx = tokenizer.encode(tokenizer.bos_token)
    config.eos_idx = tokenizer.encode(tokenizer.eos_token)

    data_loader_cls = getattr(data_loader, config.dataloader)
    dataloader = data_loader_cls(
        input_path, tokenizer, config.model.block_size
    ) if mode == 'compress' else None

    model_args = ModelArgs.from_dict(config.model)
    model_args.max_batch_size = config.model.max_batch_size
    model = GPT(model_args)
    config.model.n_params = model.n_params

    trainer_args = TrainArgs.from_dict(config.trainer)
    trainer = Trainer(trainer_args, model)

    io_mode = ''
    io_cls = getattr(file_io, config.ac.io)
    if mode == 'compress':
        io_mode += 'a'
        if 'Binary' in config.ac.io: io_mode += 'b'
        io = io_cls(output_path, io_mode)
    else:
        io_mode += 'r'
        if 'Binary' in config.ac.io: io_mode += 'b'
        io = io_cls(input_path, io_mode)

    return config, tokenizer, dataloader, trainer, io


def creat_vocab_file_with_spacy(input_file: str, output_file: str):
    import spacy
    from collections import Counter

    nlp = spacy.load("en_core_web_sm")
    with open(input_file, "r") as f:
        lines = f.readlines()
    token_list = []

    for line in tqdm(lines, desc='Tokenizing over lines'):
        doc = nlp(line.strip())
        for token in doc:
            if not token.text in token_list and len(token.text.strip()) > 0:
                token_list.append(token.text)

    random.shuffle(token_list)
    with open(output_file, "w") as f:
        for token in token_list:
            print(token, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compress a text file while training an LLM.'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help='The text file to compress.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='The output file to save the compressed text.'
    )
    args = parser.parse_args()
    creat_vocab_file_with_spacy(args.input_file, args.output_file)
