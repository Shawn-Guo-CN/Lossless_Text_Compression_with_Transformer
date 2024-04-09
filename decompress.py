import argparse
import numpy as np
import torch

from utils import set_seed, init_by_config_path


def decompress(args):
    config, tokenizer, _, trainer = init_by_config_path(
        args.input_file, args.config_file
    )
    max_len = config.model.block_size
    print('Start decompressing with the following config...')
    print(config)

    # setup global variables for compressing
    precision = config.ac.precision
    whole = 2 ** precision
    half = whole // 2
    quarter = whole // 4

    low = 0
    high = whole

    with open(args.input_file, 'r') as f:
        bits = f.read().strip()
    bits = [int(bit) for bit in bits]
    len_bits = len(bits)

    z = 0
    i = 1 # index of bits
    while i <= precision and i <= len_bits:
        if bits[i - 1] == 1:
            z = z + 2**(precision - i)
        i += 1

    output = [tokenizer.vocab[tokenizer.bos_token]]

    while output[-1] != tokenizer.eos_idx:
        # get the probabilities of the next token with ONLY FORWARD PASS
        logits = trainer.predict_step(output[-max_len:])
        probs = torch.softmax(logits[0][-1], dim=-1)
        cumprobs = torch.cumsum(probs, dim=0)
        cumprobs = torch.cat(
            (torch.tensor([0.0], device=probs.device), cumprobs), dim=0
        )

        # get the target index
        width = high - low
        assert width > 1, "Precision error."
        widths = width * cumprobs[1:] 
        lows = low + widths[:-1].int()
        highs = low + widths[1:].int()

        valid_indices = (lows < z) & (z <= highs)
        assert valid_indices.any(), "No valid index found."
        assert valid_indices.sum() == 1, "More than one valid index found."

        tgt_idx = valid_indices.nonzero(as_tuple=True)[0][0].item() + 1

        # backpropagate the target index
        _output = output + [tgt_idx]
        _out_len = min(len(output), max_len)
        _y = _output[-_out_len:]
        _loss = trainer.loss_step(logits, _y)
        trainer.optim_step(_loss)

        # update the tracker variables
        output = _output
        low = lows[tgt_idx - 1]
        high = highs[tgt_idx - 1]

        while high < half or low > half:
            if high < half:
                low = 2 * low
                high = 2 * high
                z = 2 * z
            elif low > half:
                low = 2 * (low - half)
                high = 2 * (high - half)
                z = 2 * (z - half)
            if i <= len_bits and bits[i - 1] == 1:
                z += 1
            i += 1
        while low > quarter and high < 3 * quarter:
            low = 2 * (low - quarter)
            high = 2 * (high - quarter)
            z = 2 * (z - quarter)
            if i <= len_bits and bits[i - 1] == 1:
                z += 1
            i += 1

    print('Decompression done. Writing to file...')
    with open(args.output_file, 'w') as f:
        print(' '.join([tokenizer.decode(o) for o in output]).strip(), file=f)
    print(f'Wrote to file {args.output_file}.')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Decompress a binary file while training an LLM.'
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
    parser.add_argument(
        '--config_file',
        type=str,
        help='The path to the config file.'
    )
    args = parser.parse_args()
    decompress(args)
