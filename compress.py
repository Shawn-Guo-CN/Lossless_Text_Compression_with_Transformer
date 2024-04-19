import argparse
import torch

from utils import init_by_config_path


def compress(args):
    config, _, data_loader, trainer, io = init_by_config_path(
        args.input_file, args.output_file, args.config_file, 'compress'
    )
    print('Start compressing with the following config...')
    print(config)

    # setup global variables for compressing
    precision = config.ac.precision
    whole = 2 ** precision
    half = whole // 2
    quarter = whole // 4

    low = 0
    high = whole
    s = 0

    for batch in data_loader:
        width = high - low
        if width == 1 or width == 0:
            print('Precision error when compressing')
            exit(-1)

        probs = trainer.update_step(batch)
        cumprobs = torch.cumsum(probs, dim=0)
        cumprobs = torch.cat(
            (torch.tensor([0.0], device=probs.device), cumprobs), dim=0
        )
        tgt_idx = batch['y'][-1]
        cumprob_high = cumprobs[tgt_idx + 1].to('cpu').item()
        cumprob_low = cumprobs[tgt_idx].to('cpu').item()

        high = low + int(width * cumprob_high)
        low = low + int(width * cumprob_low)

        while high < half or low > half:
            if high < half:
                io.write([0] + s*[1])
                s = 0
                low = 2 * low
                high = 2 * high
            elif low > half:
                io.write([1] + s*[0])
                s = 0
                low = 2 * (low - half)
                high = 2 * (high - half)
        while low > quarter and high < 3 * quarter:
            low = 2 * (low - quarter)
            high = 2 * (high - quarter)
            s += 1

    s += 1
    if low < quarter:
        io.write([0] + s*[1])
    else:
        io.write([1] + s*[0])

    io.close()
    print(f'Compressed to file {args.output_file}.')

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
