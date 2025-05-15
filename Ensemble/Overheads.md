# Overhead Evaluation Results

## overhead_evaluation_mlp.py with --datatype mlp
**Command:** `python overhead_evaluation_mlp.py -d mlp`

**Output:**

```
✅ Inference Time | 
Min: 2.22 ms 
 Max: 2.53 
Average: 2.25 ± 0.05 ms
✅ Peak VRAM usage: 9.25 MB
✅ FLOPs: 168960.00 FLOPs
✅ FLOPs: 0.17 MFLOPs
✅ FLOPs: 0.00 GFLOPs
✅ Total Parameters: 1,495
✅ Trainable Parameters: 1,495

```

## overhead_evaluation_mlp.py with --datatype cnn
**Command:** `python overhead_evaluation_mlp.py -d cnn`

**Output:**

```
✅ Inference Time | 
Min: 2.26 ms 
 Max: 2.54 
Average: 2.27 ± 0.03 ms
✅ Peak VRAM usage: 9.26 MB
✅ FLOPs: 187392.00 FLOPs
✅ FLOPs: 0.19 MFLOPs
✅ FLOPs: 0.00 GFLOPs
✅ Total Parameters: 1,646
✅ Trainable Parameters: 1,646

```

## overhead_evaluation_mlp.py with --datatype transformer
**Command:** `python overhead_evaluation_mlp.py -d transformer`

**Output:**

```
✅ Inference Time | 
Min: 2.28 ms 
 Max: 8.82 
Average: 2.40 ± 0.65 ms
✅ Peak VRAM usage: 9.26 MB
✅ FLOPs: 180224.00 FLOPs
✅ FLOPs: 0.18 MFLOPs
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
Min: 10.46 ms 
 Max: 13.44 
Average: 10.56 ± 0.30 ms
✅ Peak VRAM usage: 15.91 MB
✅ FLOPs: 7465984.00 GFLOPs
✅ FLOPs: 7.47 MFLOPs
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
Min: 10.68 ms 
 Max: 11.03 
Average: 10.74 ± 0.05 ms
✅ Peak VRAM usage: 28.10 MB
✅ FLOPs: 18453504.00 GFLOPs
✅ FLOPs: 18.45 MFLOPs
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
Min: 10.63 ms 
 Max: 16.30 
Average: 10.73 ± 0.56 ms
✅ Peak VRAM usage: 59.95 MB
✅ FLOPs: 48210944.00 GFLOPs
✅ FLOPs: 48.21 MFLOPs
✅ FLOPs: 0.05 GFLOPs
✅ Total Parameters: 13,586
✅ Trainable Parameters: 10,446

```

