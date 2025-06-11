import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os
import torch
from utils import read_yaml
import numpy as np

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        # Store the input and target data
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        # Returns the total number of samples
        return len(self.inputs)

    def __getitem__(self, idx):
        # Return a single sample (input, target pair)
        return self.inputs[idx], self.targets[idx]
    

def transformer_data4mlp(config, kfold=False):
    csv_file_path = os.path.join(config["data_dir"], "trans_data.csv")
    df = pd.read_csv(csv_file_path)
    df['Total_Activations_Batch_Size'] = df['Total Activations'] * df['Batch Size']
    df.loc[df['Status'] == 'OOM_CRASH', 'Max GPU Memory (MiB)'] = 42000

    df = df[['Batch Size', 'Seq Length', 'Embedding Size', 'Num Layers',
        'Num Heads', 'Depth', 'Accumulated Params', 'Activations-Params',
        'Total Activations', 'Total_Activations_Batch_Size', 'Total Parameters', 'Max GPU Memory (MiB)',
        'NonDynamicallyQuantizableLinear Count', 'Linear Count',
        'LayerNorm Count', 'Dropout Count', 'Status']]
    
    bins = [0, 8000, 16000, 24000, 32000, 40000, 50000]
    labels = [i for i in range(0, len(bins) - 1)]

    # Create the 'memory_usage_label' column
    df['memory_usage_label'] = pd.cut(df['Max GPU Memory (MiB)'], bins=bins, labels=labels, right=False)

    # Count instances per class
    class_counts = len(df['memory_usage_label'].value_counts())
    desired_features = df[['Depth','Total Activations', 'Total_Activations_Batch_Size', 'Total Parameters', 'Batch Size',
                      'NonDynamicallyQuantizableLinear Count', 'Linear Count',
                      'LayerNorm Count', 'Dropout Count']].values
    desired_labels = df['memory_usage_label'].values
    if not kfold:
        x_train, x_temp, y_train, y_temp = train_test_split(desired_features, desired_labels, test_size=0.3, random_state=42)

        x_test, x_val, y_test, y_val = train_test_split(
            x_temp, y_temp, test_size=0.66, random_state=42)
        
        train_dataloader = DataLoader(CustomDataset(x_train, y_train), config["batch_size"], num_workers=2, shuffle=True)
        val_dataloader = DataLoader(CustomDataset(x_val, y_val), config["batch_size"], num_workers=2)
        test_dataloader = DataLoader(CustomDataset(x_test, y_test), 1, num_workers=1)
        return train_dataloader, val_dataloader, test_dataloader, class_counts
    else:
        whole_dataset = CustomDataset(desired_features, desired_labels)
        return whole_dataset, class_counts



def cnn_data4mlp(config, kfold=False):
    csv_file_path = os.path.join(config["data_dir"], "cnn_data1.csv")
    df = pd.read_csv(csv_file_path)
    df['Total_Activations_Batch_Size'] = df['Total Activations'] * df['Batch Size']
    df.loc[df['Status'] == 'OOM_CRASH', 'Max GPU Memory (MiB)'] = 42000
    df = df.dropna(subset=['Activation Function'])
    df = df[~df['architecture'].isin(['residual', 'dense'])]

    df = df[['Depth', 'Activation Function', 'Total Activations', 'Total_Activations_Batch_Size', 'Total Parameters', 'Batch Size',
                'Conv2d Count','BatchNorm2d Count', 'Dropout Count', 'AdaptiveAvgPool2d Count',
                'Linear Count', 'Max GPU Memory (MiB)', 'architecture',
                'Input Size (MB)', 'Forward/Backward Pass Size (MB)','Params Size (MB)', 'Estimated Total Size (MB)']]

    # List of activation functions
    activations = ['ELU', 'GELU', 'LeakyReLU', 'Mish', 'PReLU', 'ReLU', 'SELU', 'SiLU', 'Softplus', 'Tanh']

    # Function to create positional encoding
    def positional_encoding_2d(num_states):
        positions = []
        for i in range(num_states):
            position = (np.sin(i * np.pi / num_states), np.cos(i * np.pi / num_states))
            positions.append(position)
        return np.array(positions)

    # Generate positional encodings
    positional_encodings = positional_encoding_2d(len(activations))
    activation_to_encoding = {activation: positional_encodings[i] for i, activation in enumerate(activations)}

    # Apply positional encoding to 'activation_function' column
    df['activation_encoding_sin'] = df['Activation Function'].map(lambda x: activation_to_encoding[x][0])
    df['activation_encoding_cos'] = df['Activation Function'].map(lambda x: activation_to_encoding[x][1])
    bins = [0, 8000, 16000, 24000, 32000, 40000, 50000]
    labels = [i for i in range(0, len(bins) - 1)]

    # Create the 'memory_usage_label' column
    df['memory_usage_label'] = pd.cut(df['Max GPU Memory (MiB)'], bins=bins, labels=labels, right=False)

    # Count instances per class
    class_counts = len(df['memory_usage_label'].value_counts())
    desired_features = df[['Depth', 'Total Activations', 'Total_Activations_Batch_Size', 'Total Parameters', 'Batch Size',
            'Conv2d Count', 'BatchNorm2d Count', 'Dropout Count', 'activation_encoding_sin',
            'activation_encoding_cos',
            # 'Input Size (MB)', 'Forward/Backward Pass Size (MB)','Params Size (MB)', 'Estimated Total Size (MB)',
            ]].values
    desired_labels = df['memory_usage_label'].values 
    if not kfold:
        x_train, x_temp, y_train, y_temp = train_test_split(desired_features, desired_labels, test_size=0.3, random_state=42)

        x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.66, random_state=42)
        train_dataloader = DataLoader(CustomDataset(x_train, y_train), config["batch_size"], num_workers=2, shuffle=True)
        val_dataloader = DataLoader(CustomDataset(x_val, y_val), config["batch_size"], num_workers=2)
        test_dataloader = DataLoader(CustomDataset(x_test, y_test),1, num_workers=1)
        return train_dataloader, val_dataloader, test_dataloader, class_counts
    else:
        whole_dataset = CustomDataset(desired_features, desired_labels)
        return whole_dataset, class_counts

def mlp_data4mlp(config, kfold=False):
    csv_file_path = os.path.join(config["data_dir"],"mlp_data2.csv")
    # Load the CSV into a DataFrame and assign it to the desired column names
    df = pd.read_csv(csv_file_path)
    df = df[['Max GPU Memory (MiB)', 'Depth', 'Activation Function', 'Batch Size', 'Batch Normalization Layers', 'Dropout Layers', 'Total Parameters', 'Total Activations', 'Activations-Params', ]]

    # Map the existing columns to your desired column structure
    df = df.rename(columns={
        'Max GPU Memory (MiB)': 'real_memory_usage',
        'Depth': 'layers',
        'Activation Function': 'activation_function',
        'Batch Size': 'batch_size',
        'Total Parameters': 'all_parameters',
        'Total Activations': 'all_activations',
        'Activations-Params': 'params_neurons_list',
        'Batch Normalization Layers': 'batch_norm_layer',
        'Dropout Layers': 'dropout_layers'
    })
    # List of activation functions
    activations = ['ELU', 'GELU', 'Identity', 'LeakyReLU', 'Mish', 'PReLU', 'ReLU', 'SELU', 'SiLU', 'Softplus', 'Tanh']

    # Function to create positional encoding
    def positional_encoding_2d(num_states):
        positions = []
        for i in range(num_states):
            position = (np.sin(i * np.pi / num_states), np.cos(i * np.pi / num_states))
            positions.append(position)
        return np.array(positions)

    # Generate positional encodings
    positional_encodings = positional_encoding_2d(len(activations))
    activation_to_encoding = {activation: positional_encodings[i] for i, activation in enumerate(activations)}

    # Apply positional encoding to 'activation_function' column
    df['activation_encoding_sin'] = df['activation_function'].map(lambda x: activation_to_encoding[x][0])
    df['activation_encoding_cos'] = df['activation_function'].map(lambda x: activation_to_encoding[x][1])

    bins = [i*1000 for i in range(1, 6)] 
    bins.append(float('inf'))
    labels = [i for i in range(len(bins)-1)]  
    class_counts = len(labels)

    df['memory_usage_label'] = pd.cut(df['real_memory_usage'], bins=bins, labels=labels, right=True)
    desired_features = df[['layers', 'batch_size', 'all_parameters', 'all_activations', 'batch_norm_layer', 'dropout_layers', 'activation_encoding_sin', 'activation_encoding_cos']].values
    desired_labels = df['memory_usage_label'].values
    if not kfold:
        x_train, x_temp, y_train, y_temp = train_test_split(desired_features, desired_labels, test_size=0.3, random_state=42)

        x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.66, random_state=42)

        train_dataloader = DataLoader(CustomDataset(x_train, y_train), config["batch_size"], num_workers=2, shuffle=True)
        val_dataloader = DataLoader(CustomDataset(x_val, y_val), config["batch_size"], num_workers=2)
        test_dataloader = DataLoader(CustomDataset(x_test, y_test),1, num_workers=1)

        return train_dataloader, val_dataloader, test_dataloader, class_counts
    else:
        whole_dataset = CustomDataset(desired_features, desired_labels)
        return whole_dataset, class_counts

if __name__ == "__main__":
    config = read_yaml("config.yaml")
    train_dataloader, val_dataloader, test_dataloader, class_counts = transformer_data4mlp(config)
    x, y = next(iter(train_dataloader))
    print(f"input shape: {x.shape}, output shape: {y.shape}")
    print(f"number of classes: {class_counts}")
