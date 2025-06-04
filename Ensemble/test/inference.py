import torch

import sys
import os

# Get the parent directory (i.e., one level up)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


from utils import read_yaml
from models.transformer_models import TransformerEnsemble
from models.mlp_models import EnsembleModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Configuration loading
def _get_config():
    return read_yaml("../config.yaml")

def _get_architectures():
    return read_yaml("transformers_architectures.yaml")["model_configs"]

# MLP Inference Wrappers

def infer_mlp4transformer(input_list):
    config = _get_config()
    model = EnsembleModel.load_from_checkpoint(
        "../trained_models/mlp4transformer/mlp4transformer.ckpt",
        model_list=[1, 2, 3, 4, 5, 6],
        input_size=len(input_list),
        output_size=6,
        max_neurons=8,
        min_neurons=4,
        learning_rate=config["learning_rate"]
    )
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(input_list, dtype=torch.float32).unsqueeze(0)
        logits = model(tensor)
        return logits.argmax(dim=1).item()

def infer_mlp4cnn(input_list):
    config = _get_config()
    model = EnsembleModel.load_from_checkpoint(
        "../trained_models/mlp4cnn/mlp4cnn.ckpt",
        model_list=[1, 2, 3, 4, 5, 6],
        input_size=len(input_list),
        output_size=6,
        max_neurons=8,
        min_neurons=4,
        learning_rate=config["learning_rate"]
    )
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(input_list, dtype=torch.float32).unsqueeze(0)
        logits = model(tensor)
        return logits.argmax(dim=1).item()

def infer_mlp4mlp(input_list):
    config = _get_config()
    model = EnsembleModel.load_from_checkpoint(
        "../trained_models/mlp4mlp/mlp4mlp.ckpt",
        model_list=[1,2,3,4,5,6,7,1,2,3,4,5,6],
        input_size=len(input_list),
        output_size=5,
        max_neurons=8,
        min_neurons=4,
        learning_rate=config["learning_rate"]
    )

    model.eval()
    tensor = torch.tensor(input_list, dtype=torch.float32).unsqueeze(0).to(device)


    with torch.no_grad():
        tensor = torch.tensor(input_list, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(tensor)
        return logits.argmax(dim=1).item()

# Transformer Inference Wrappers

def infer_transformer4transformer(sequence_list, extra_list):
    config = _get_config()
    architectures = _get_architectures()
    model = TransformerEnsemble.load_from_checkpoint(
        "../trained_models/transformer4transformer/transformer4transformer.ckpt",
        model_configs=architectures,
        num_features=len(sequence_list[0]),
        num_classes=6,
        learning_rate=config["learning_rate"],
        max_seq_len=40,
        extra_fetures_num=len(extra_list)
    )
    model.eval()
    with torch.no_grad():
        seq_tensor = torch.tensor([sequence_list], dtype=torch.float32)
        extra_tensor = torch.tensor([extra_list], dtype=torch.float32)
        logits = model(seq_tensor, extra_tensor)
        return logits.argmax(dim=1).item()

def infer_transformer4cnn(sequence_list, extra_list):
    config = _get_config()
    architectures = _get_architectures()
    model = TransformerEnsemble.load_from_checkpoint(
        "../trained_models/transformer4cnn/transformer4cnn.ckpt",
        model_configs=architectures,
        num_features=len(sequence_list[0]),
        num_classes=6,
        learning_rate=config["learning_rate"],
        max_seq_len=35,
        extra_fetures_num=len(extra_list)
    )
    model.eval()
    with torch.no_grad():
        seq_tensor = torch.tensor([sequence_list], dtype=torch.float32)
        extra_tensor = torch.tensor([extra_list], dtype=torch.float32)
        logits = model(seq_tensor, extra_tensor)
        return logits.argmax(dim=1).item()

def infer_transformer4mlp(sequence_list, extra_list):
    config = _get_config()
    architectures = _get_architectures()
    model = TransformerEnsemble.load_from_checkpoint(
        "../trained_models/transformer4mlp/transformer4mlp.ckpt",
        model_configs=architectures,
        num_features=len(sequence_list[0]),
        num_classes=5,
        learning_rate=config["learning_rate"],
        max_seq_len=30,
        extra_fetures_num=len(extra_list)
    )
    model.eval()
    with torch.no_grad():
        seq_tensor = torch.tensor([sequence_list], dtype=torch.float32)
        extra_tensor = torch.tensor([extra_list], dtype=torch.float32)
        logits = model(seq_tensor, extra_tensor)
        return logits.argmax(dim=1).item()


print("gashang tar az fereshte has!")

# 'layers', 'batch_size', 'all_parameters', 'all_activations', 'batch_norm_layer', 'dropout_layers', 'activation_encoding_sin', 'activation_encoding_cos'

print(infer_mlp4mlp([100, 100, 1092830981, 290374098273, 1234, 123123, 0.1, 0.2]))