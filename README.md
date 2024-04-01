# Lossless_Text_Compression_with_Transformer

This repo is to demo the concept of lossless compression with Transformers as encoder and decoder.

The modules are:

1. `compress.py`: The bash script to compress a text file with the Transformer model
2. `data_loader.py`: The DataLoader class for loading the text file
3. `decompress.py`: The bash script to decompress a compressed file with the Transformer model
4. `model.py`: The Transformer model
5. `trainer.py`: The Trainer class for updating the model parameters
6. `tokenizer.py`: The Tokenizer class
7. `utils.py`: Utility functions

## Usage

## Compress

```bash
python compress.py --input_file <input_file> --output_file <output_file> --config_file <config_file>
```

### Arguments

- `input_file`: The path to the input text file, e.g. `data/demo.txt`
- `output_file`: The path to the output compressed file, e.g. `data/demo_encode_out.txt`
- `config_file`: The path to the configuration file in the YAML format, e.g. `config/global/demo.yaml`


### Pipeline of the compression

1. Tokenize the input text
2. Calculate the probability of a token given the previous tokens by the forward pass of Transformer model
3. Encode the token with the probability and the arithmetic coding algorithm
4. Output the arithmetic code to a text file for readability

## Decompress

```bash
python decompress.py --input_file <input_file> --output_file <output_file> --config_file <config_file>
```

### Arguments

- `input_file`: The path to the input text file, e.g. `data/demo_encode_out.txt`
- `output_file`: The path to the output compressed file, e.g. `data/demo_decode_out.txt`
- `config_file`: The path to the configuration file in the YAML format, e.g. `config/global/demo.yaml`


### Pipeline of the decompression

1. Read the arithmetic code from the input file
2. Decode the arithmetic code to the token while getting probability from Transformer model and updating the parameters of the model
3. Detokenise the tokens
4. Output the detokenised text to the output file

