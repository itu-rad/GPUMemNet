# Overhead Evaluation Results

## overhead_evaluation_mlp.py with --datatype mlp
**Command:** `python overhead_evaluation_mlp.py -d mlp`

**Output:**

```

🧠 GPU Inference Stats:
  Min:     2.34 ms
  Max:     2.61 ms
  Avg:     2.36 ± 0.03 ms
  VRAM:    9.25 MB

🧠 CPU Inference Stats:
  Min:     1.87 ms
  Max:     2.06 ms
  Avg:     1.90 ± 0.04 ms

✅ FLOPs: 168960.00 (0.00 GFLOPs)

✅ Parameters: 1,495 total | 1,495 trainable

```

## overhead_evaluation_mlp.py with --datatype cnn
**Command:** `python overhead_evaluation_mlp.py -d cnn`

**Output:**

```

🧠 GPU Inference Stats:
  Min:     2.35 ms
  Max:     2.69 ms
  Avg:     2.38 ± 0.03 ms
  VRAM:    9.26 MB

🧠 CPU Inference Stats:
  Min:     1.89 ms
  Max:     2.79 ms
  Avg:     2.00 ± 0.21 ms

✅ FLOPs: 187392.00 (0.00 GFLOPs)

✅ Parameters: 1,646 total | 1,646 trainable

```

## overhead_evaluation_mlp.py with --datatype transformer
**Command:** `python overhead_evaluation_mlp.py -d transformer`

**Output:**

```

🧠 GPU Inference Stats:
  Min:     2.33 ms
  Max:     2.65 ms
  Avg:     2.39 ± 0.05 ms
  VRAM:    9.26 MB

🧠 CPU Inference Stats:
  Min:     1.91 ms
  Max:     10.28 ms
  Avg:     2.03 ± 0.83 ms

✅ FLOPs: 180224.00 (0.00 GFLOPs)

✅ Parameters: 1,590 total | 1,590 trainable

```

## overhead_evaluation_transformers.py with --datatype mlp
**Command:** `python overhead_evaluation_transformers.py -d mlp`

**Output:**

```
Maximum layers: 46

🧠 GPU Inference Stats:
  Min:     11.03 ms
  Max:     12.10 ms
  Avg:     11.14 ± 0.15 ms
  VRAM:    15.91 MB

🧠 CPU Inference Stats:
  Min:     15.37 ms
  Max:     17.04 ms
  Avg:     15.65 ± 0.18 ms

✅ FLOPs: 7465984.00 (0.01 GFLOPs)

✅ Parameters: 7,556 total | 7,096 trainable

```

## overhead_evaluation_transformers.py with --datatype cnn
**Command:** `python overhead_evaluation_transformers.py -d cnn`

**Output:**

```
['GELU' 'SELU' 'ELU' 'ReLU' 'Mish' 'Softplus' 'Tanh' 'PReLU' 'LeakyReLU'
 'SiLU']
Maximum layers: 119

🧠 GPU Inference Stats:
  Min:     10.82 ms
  Max:     11.51 ms
  Avg:     10.96 ± 0.09 ms
  VRAM:    28.10 MB

🧠 CPU Inference Stats:
  Min:     16.72 ms
  Max:     26.17 ms
  Avg:     20.30 ± 2.06 ms

✅ FLOPs: 18453504.00 (0.02 GFLOPs)

✅ Parameters: 10,056 total | 8,866 trainable

```

## overhead_evaluation_transformers.py with --datatype transformer
**Command:** `python overhead_evaluation_transformers.py -d transformer`

**Output:**

```
Maximum layers: 314

🧠 GPU Inference Stats:
  Min:     10.79 ms
  Max:     11.89 ms
  Avg:     11.35 ± 0.38 ms
  VRAM:    59.95 MB

🧠 CPU Inference Stats:
  Min:     23.91 ms
  Max:     31.83 ms
  Avg:     25.22 ± 0.83 ms

✅ FLOPs: 48210944.00 (0.05 GFLOPs)

✅ Parameters: 13,586 total | 10,446 trainable

```

