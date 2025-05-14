# Overhead Evaluation Results

## overhead_evaluation_mlp.py with --datatype mlp
**Command:** `python overhead_evaluation_mlp.py -d mlp`

**Output:**

```
✅ Inference Time | 
Min: 2.27 ms 
 Max: 8.60 
Average: 2.37 ± 0.63 ms
✅ Peak VRAM usage: 9.25 MB
✅ FLOPs: 0.00 GFLOPs
✅ Total Parameters: 1,495
✅ Trainable Parameters: 1,495

```

## overhead_evaluation_mlp.py with --datatype cnn
**Command:** `python overhead_evaluation_mlp.py -d cnn`

**Output:**

```
✅ Inference Time | 
Min: 2.25 ms 
 Max: 3.21 
Average: 2.32 ± 0.10 ms
✅ Peak VRAM usage: 9.26 MB
✅ FLOPs: 0.00 GFLOPs
✅ Total Parameters: 1,646
✅ Trainable Parameters: 1,646

```

## overhead_evaluation_mlp.py with --datatype transformer
**Command:** `python overhead_evaluation_mlp.py -d transformer`

**Output:**

```
✅ Inference Time | 
Min: 2.26 ms 
 Max: 3.14 
Average: 2.29 ± 0.09 ms
✅ Peak VRAM usage: 9.26 MB
✅ FLOPs: 0.00 GFLOPs
✅ Total Parameters: 1,590
✅ Trainable Parameters: 1,590

```

## overhead_evaluation_transformers.py with --datatype mlp
**Command:** `python overhead_evaluation_transformers.py -d mlp`

**Output:**

```
Maximum layers: 46
✅ Inference Time | 
Min: 10.50 ms 
 Max: 14.22 
Average: 10.70 ± 0.36 ms
✅ Peak VRAM usage: 15.91 MB
✅ FLOPs: 0.01 GFLOPs
✅ Total Parameters: 7,556
✅ Trainable Parameters: 7,096

```

## overhead_evaluation_transformers.py with --datatype cnn
**Command:** `python overhead_evaluation_transformers.py -d cnn`

**Output:**

```
['GELU' 'SELU' 'ELU' 'ReLU' 'Mish' 'Softplus' 'Tanh' 'PReLU' 'LeakyReLU'
 'SiLU']
Maximum layers: 119
✅ Inference Time | 
Min: 10.52 ms 
 Max: 13.08 
Average: 10.69 ± 0.27 ms
✅ Peak VRAM usage: 28.10 MB
✅ FLOPs: 0.02 GFLOPs
✅ Total Parameters: 10,056
✅ Trainable Parameters: 8,866

```

## overhead_evaluation_transformers.py with --datatype transformer
**Command:** `python overhead_evaluation_transformers.py -d transformer`

**Output:**

```
Maximum layers: 314
✅ Inference Time | 
Min: 10.56 ms 
 Max: 20.99 
Average: 10.72 ± 1.03 ms
✅ Peak VRAM usage: 59.95 MB
✅ FLOPs: 0.05 GFLOPs
✅ Total Parameters: 13,586
✅ Trainable Parameters: 10,446

```

