# Overhead Evaluation Results

## overhead_evaluation_mlp.py with --datatype mlp
**Command:** `python overhead_evaluation_mlp.py -d mlp`

**Output:**

```
✅ Inference Time: 2.36 ± 0.68 ms
✅ Peak VRAM usage: 9.25 MB
✅ FLOPs: 0.00 GFLOPs
✅ Total Parameters: 1,495
✅ Trainable Parameters: 1,495

```

## overhead_evaluation_mlp.py with --datatype cnn
**Command:** `python overhead_evaluation_mlp.py -d cnn`

**Output:**

```
✅ Inference Time: 2.29 ± 0.09 ms
✅ Peak VRAM usage: 9.26 MB
✅ FLOPs: 0.00 GFLOPs
✅ Total Parameters: 1,646
✅ Trainable Parameters: 1,646

```

## overhead_evaluation_mlp.py with --datatype transformer
**Command:** `python overhead_evaluation_mlp.py -d transformer`

**Output:**

```
✅ Inference Time: 2.37 ± 0.71 ms
✅ Peak VRAM usage: 9.26 MB
✅ FLOPs: 0.00 GFLOPs
✅ Total Parameters: 1,590
✅ Trainable Parameters: 1,590

```

## overhead_evaluation_transformers.py with --datatype mlp
**Command:** `python overhead_evaluation_transformers.py -d mlp`

**Output:**

```
✅ Inference Time: 10.77 ± 0.33 ms
✅ Peak VRAM usage: 15.91 MB
✅ FLOPs: 0.01 GFLOPs
✅ Total Parameters: 7,556
✅ Trainable Parameters: 7,096

```

## overhead_evaluation_transformers.py with --datatype cnn
**Command:** `python overhead_evaluation_transformers.py -d cnn`

**Output:**

```
✅ Inference Time: 10.77 ± 0.19 ms
✅ Peak VRAM usage: 28.10 MB
✅ FLOPs: 0.02 GFLOPs
✅ Total Parameters: 10,056
✅ Trainable Parameters: 8,866

```

## overhead_evaluation_transformers.py with --datatype transformer
**Command:** `python overhead_evaluation_transformers.py -d transformer`

**Output:**

```
✅ Inference Time: 10.73 ± 0.42 ms
✅ Peak VRAM usage: 59.95 MB
✅ FLOPs: 0.05 GFLOPs
✅ Total Parameters: 13,586
✅ Trainable Parameters: 10,446

```

