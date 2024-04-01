import argparse
import numpy as np

from utils import set_seed, init_by_config_path


def decompress(args):
    config, tokenizer, _, trainer = init_by_config_path(
        args.input_file, args.config_file
    )
    max_len = config.model.block_size

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

    output = [0]
    while output[-1] != tokenizer.eos_idx:
        probs = trainer.predict_step(output[-max_len:]).to('cpu').numpy()
        cumprobs = np.cumsum(probs)
        cumprobs = np.insert(cumprobs, 0, 0.)

        for tgt_idx in range(1, tokenizer.vocab_size):
            width = high - low
            if width == 1 or width == 0:
                print("precision error")

            high_ = low + int(width * cumprobs[tgt_idx + 1])
            low_ = low + int(width * cumprobs[tgt_idx])

            if low_ <= z < high_:
                _output = output + [tgt_idx]
                print(tokenizer.decode(tgt_idx))

                _out_len = min(len(output), max_len)
                _batch = {
                    'x': output[-_out_len:],
                    'y': _output[-_out_len:]
                }
                _ = trainer.step(_batch)

                output = _output
                low = low_
                high = high_
                break

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

    with open(args.output_file, 'w') as f:
        print(' '.join([tokenizer.decode(o) for o in output]), file=f)

    return


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
    parser.add_argument(
        '--config_file',
        type=str,
        help='The path to the config file.'
    )
    args = parser.parse_args()
    decompress(args)
