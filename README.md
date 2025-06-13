# GPUMemNet: GPU Memory estimator and Neural Network training dataset
<p align="center">
  <img src="image/logo_with_background.png" alt="Alt Text" width="200"/>
</p>

This repository contains the artifacts for our work on building a deep learning–based GPU memory estimator for training deep learning models. Since data is central to this effort, we structured the workflow in several key stages:

-	**Data Generation**: We developed scripts to automatically generate diverse deep learning training configurations and monitor GPU behavior during training.

-	**Data Cleaning**: After collecting raw logs, we processed and cleaned the data using dedicated scripts included here.

-	**Analysis & Modeling**: With the cleaned data, we performed exploratory analysis and trained various models to estimate GPU memory usage.

-	We explored **ensemble method**, reviewed **related work**, and analyzed the **overhead** introduced by both the data parsers and model inference.
 


## Related Work data and sources
We designed experiments to evaluate the effectiveness of the Horus formula and the Fake Tensor library in estimating the GPU memory requirements of deep learning training tasks. Read more [here](Related_works/README.md).



## Data Generation Scripts
For each neural network type (**MLP**, **CNN**, **Transformer**), we provide two key files: one that defines the network architecture, and a launcher script that spawns multiple training instances with varying architectural parameters. During training, GPU usage (alongisde with other metrics) is monitored using dcgmi and nvidia-smi, while system metrics are tracked with top. 

**Note 1**: Each deep learning configuration is trained for one minute, one at a time. This sequential execution avoids interference from simultaneous training jobs, which could affect system performance due to shared CPU and DRAM usage.

- MLP: [MLP model](NeuroNetGen_scripts/MLP/mlp_next.py) |  [MLP model launcher](NeuroNetGen_scripts/MLP/launcher_mlp_next.py)
- CNN: [CNN model](NeuroNetGen_scripts/CNN/cnn_with_model_summary.py) | [CNN model launcher](NeuroNetGen_scripts/CNN/launcher_cnn_with_modelsummary.py)
- Transformer: [Transformer model](NeuroNetGen_scripts/Transformer/transformer.py) | [Transformer model launcher](NeuroNetGen_scripts/Transformer/launcher_transformer.py)

### Future/ Possible Contributions at This Level

1. Refactoring the launcher script to read parameters from a YAML configuration file.
2. Extending the Transformer model to support architectures with 1D convolutional layers (e.g., GPT-style models), as it currently supports only linear-layer-based designs.

## Data Cleaning Script
- [MLP cleaner](Data_Cleaner_scripts/MLP/data_cleaner_mlp_step2.py)
- [CNN cleaner](Data_Cleaner_scripts/CNN/data_cleaner_cnn_step1.py)
- [Transformers cleaner](Data_Cleaner_scripts/Transformer/data_cleaner_transformer_m2.py)

### Future/ Possible Contributions at This Level
- Extend the Transformer data cleaning script to support models that include Conv1D and other types of layers


## Data
- [MLPs](Datasets/MLP/mlp_data_step2.csv)
- [CNNs](Datasets/CNN/cnn_data_step1.csv)
- [Transformers](Datasets/Transformers/transformer_data.csv)


## Visualization, Analysis, and Training Notebooks
We looked into the cleaned data by looking into its distribution based on different selected features, visualized through PCA and TSNE glasses. Also, trained MLP-, and transformer-based models on then to validate the idea of using deep learning for estimating GPU memory usage. For diving into this check more [here](Analysis/00-Cleaned-NoteBooks/README.md).

### Training, Validation, and Testing with Ensemble Models

To train and test ensemble models, ensure that you are using the correct dataset. When running training or evaluation, specify both the dataset and the model type using the appropriate command-line arguments.

**Training:**
```bash
python train.py --d [mlp, cnn, transformer] --m [mlp, transformer]
```

**Validation:**
```bash
python kfold_cross_validation.py --d [mlp, cnn, transformer] --m [mlp, transformer]
```

**Training:**
```bash
python test.py --d [mlp, cnn, transformer] --m [mlp, transformer]
```

To visualize the results, including the confusion matrix and other statistics, see the [visualization notebook](Ensemble/visualize_results.ipynb).

## Overheads of the parser and the models' inference
We also considered and characterized the overheads of parsers and the estimator models' overhead since one of the primary purpose of these estimators can be informing schedules/ resource managers to make more efficient decisions. 
- [Parser overhead](Test_Pipeline/parser_overhead.md)
- [Models' inference overhead](Ensemble/Overheads.md)



## Vision
In the discussion section of our paper, we draw the roadmap on how contributors can contribute. As it is an deep learning-based estimator, the potential contributions and improvements to the current study can come from more data points, data points from different GPU models, with broader range of arguments, and also innovations on how to view the GPU memory estimation.