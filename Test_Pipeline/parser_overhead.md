# Parser Overhead Report

Ran each parser 100 times

## CNN/MLP Parser

**Input:** ('cnn_models/efficientnet_32.model', 32)
- Min: 0.96 ms
- Max: 2.04 ms
- Mean: 0.99 ms
- Std: 0.14 ms
- Mean ± Std: 0.99 ± 0.14 ms

## Transformer/MLP Parser

**Input:** ('Trans_models/gpt2_xl_bs:2_sl:512.txt', 32, 512)
- Min: 1.74 ms
- Max: 2.60 ms
- Mean: 1.77 ms
- Std: 0.09 ms
- Mean ± Std: 1.77 ± 0.09 ms
