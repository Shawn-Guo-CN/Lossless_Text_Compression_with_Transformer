# Lossless_Text_Compression_with_Transformer

This repo is to demo the concept of lossless compression with Transformers as encoder and decoder.

The modules are:

1. `compress.py`: The bash script to compress a text file with the Transformer model
2. `decompress.py`: The bash script to decompress a compressed file with the Transformer model
3. `model.py`: The Transformer model
4. `tokenizer.py`: The Tokenizer class
5. `utils.py`: Utility functions

## Usage

## Compress

```bash
python compress.py --input_file <input_file> --output_file <output_file> --model_config <model_path> --tokenizer_config <tokenizer_path>
```

Pipeline for compression:

1. Tokenize the input text
2. Calculate the probability of a token given the previous tokens by the forward pass of Transformer model
3. Encode the token with the probability by arithmetic coding
4. Output the arithmetic code to the output file in binary format

## Decompress

```bash
python decompress.py --input_file <input_file> --output_file <output_file> --model_config <model_path> --tokenizer_config <tokenizer_path>
```

Pipeline for decompression:

1. Read the arithmetic code from the input file
2. Decode the arithmetic code to the token with the probability by Transformer model
3. Detokenise the tokens
4. Output the detokenised text to the output file

