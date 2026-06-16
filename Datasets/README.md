# GPUMemNet Datasets

This directory contains datasets collected for estimating GPU memory
consumption and utilization characteristics of deep learning training
workloads.

The files represent different workload families and data-generation stages.
They do not share a single common schema and should not be concatenated
without explicit preprocessing.

## Dataset files

| File | Workload family | Rows | Description |
|---|---:|---:|---|
| `MLP/mlp_data_step1.csv` | MLP | 3,000 | MLP configurations without explicit batch-normalization and dropout counts |
| `MLP/mlp_data_step2.csv` | MLP | 3,000 | MLP configurations including batch-normalization and dropout counts |
| `CNN/cnn_data_step1.csv` | CNN | 9,000 | CNN configurations including an architecture identifier |
| `CNN/cnn_data_new_approach.csv` | CNN | 9,000 | Revised CNN representation without the architecture identifier |
| `Transformers/transformer_data.csv` | Transformer | 5,011 | Transformer configurations and recorded execution status |

## Prediction target

The primary GPU-memory prediction target is:

- `Max GPU Memory (MiB)`

## Utilization metrics

The datasets in this GitHub repository include average values for:

- `Avg GPUTL`
- `Avg GRACT`
- `Avg SMACT`
- `Avg SMOCC`
- `Avg FP32A`

The corresponding Hugging Face release additionally provides
utilization-focused variants for MLP, CNN, and Transformer workloads.

These variants contain average and maximum values for:

- `GPUTL`
- `GRACT`
- `SMACT`
- `SMOCC`
- `FP32A`
- `DRAMA`

The exact columns differ across workload families and dataset variants.

## Workload-specific features

The datasets use different features for each workload family:

- MLP data includes depth, activation function, activation counts, parameter
  counts, batch size, and optionally batch-normalization and dropout counts.
- CNN data additionally includes layer-type counts and analytical memory-size
  estimates.
- Transformer data includes sequence length, embedding size, number of
  attention heads, number of layers, and transformer-specific layer counts.

Column names and units are preserved from the original experimental pipeline
for compatibility and reproducibility.

## Release relationship

The files stored in this GitHub repository are byte-for-byte identical to
their corresponding files in the Hugging Face release:

| GitHub file | Hugging Face file |
|---|---|
| `MLP/mlp_data_step1.csv` | `MLP/mlp_data1.csv` |
| `MLP/mlp_data_step2.csv` | `MLP/mlp_data2.csv` |
| `CNN/cnn_data_step1.csv` | `CNN/cnn_data1.csv` |
| `CNN/cnn_data_new_approach.csv` | `CNN/cnn_data_new_approach.csv` |
| `Transformers/transformer_data.csv` | `Transformers/transformer_data1.csv` |

The Hugging Face release additionally contains:

- utilization-focused datasets with average and maximum telemetry metrics;
- an earlier fully connected network GPU-memory dataset.

The canonical public dataset release is available at:

https://huggingface.co/datasets/ehyo/GPU-Resources-Estimation-for-Deep-Learning-Training-Tasks

The copies in this repository are retained for compatibility with the
original code and published experiments.

## Loading the datasets

Load each file independently:

~~~~python
import pandas as pd

mlp = pd.read_csv("Datasets/MLP/mlp_data_step2.csv")
cnn = pd.read_csv("Datasets/CNN/cnn_data_new_approach.csv")
transformers = pd.read_csv(
    "Datasets/Transformers/transformer_data.csv"
)
~~~~

Do not automatically concatenate these files because their schemas and
workload-specific features differ.

## License

These datasets are licensed under the Creative Commons Attribution 4.0
International License unless otherwise stated.

See the repository-level `README.md` and `NOTICE` files for licensing and
attribution details.
