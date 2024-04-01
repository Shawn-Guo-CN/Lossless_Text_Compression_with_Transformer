import argparse
import numpy as np

from utils import set_seed, init_by_config_path


def compress(args):
    config, _, data_loader, trainer = init_by_config_path(
        args.input_file, args.config_file
    )

    # setup global variables for compressing
    precision = config.ac.precision
    whole = 2 ** precision
    half = whole // 2
    quarter = whole // 4

    low = 0
    high = whole
    s = 0

    bits = []

    for batch in data_loader:
        width = high - low
        if width == 1 or width == 0:
            print("precision error")

        probs = trainer.step(batch).to('cpu').numpy()
        cumprobs = np.cumsum(probs)
        cumprobs = np.insert(cumprobs, 0, 0.)
        tgt_idx = batch['y'][-1]

        high = low + int(width * cumprobs[tgt_idx + 1])
        low = low + int(width * cumprobs[tgt_idx])

        while high < half or low > half:
            if high < half:
                bits += [0] + s*[1]
                s = 0
                low = 2 * low
                high = 2 * high
            elif low > half:
                bits += [1] + s*[0]
                s = 0
                low = 2 * (low - half)
                high = 2 * (high - half)
        while low > quarter and high < 3 * quarter:
            low = 2 * (low - quarter)
            high = 2 * (high - quarter)
            s += 1

    s += 1
    if low < quarter:
        bits += [0] + s*[1]
    else:
        bits += [1] + s*[0]

    with open(args.output_file, 'w') as f:
        print(''.join([str(bit) for bit in bits]), file=f)

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
    compress(args)
