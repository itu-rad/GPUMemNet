import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch 
import os
import ast


class LayerSequenceDataset(Dataset):
    def __init__(self, x_data, y_labels, max_seq_len, data_type='transformer'):
        self.x_data = [process_sequence_data(seq[0], data_type=data_type) for seq in x_data]
        self.batch_size = [np.array(seq[1:]).astype(np.int64) for seq in x_data]
        self.y_labels = y_labels
        self.x_data, self.batch_size, self.y_labels = get_filtered_lists(self.x_data, self.batch_size, self.y_labels)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        sequence = self.x_data[idx]
        batch_size = self.batch_size[idx]
        label = self.y_labels[idx]
        # Padding sequences to max length
        if len(sequence) < self.max_seq_len:
            padded_sequence = np.pad(sequence, ((0, self.max_seq_len - len(sequence)), (0,0)), 'constant')
        else:
            padded_sequence = sequence[:self.max_seq_len]
        return torch.tensor(padded_sequence, dtype=torch.float32), torch.tensor(batch_size, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def encode_layer_transformer(layer_type):
    if layer_type in 'LayerNorm':
        return [1, 0, 0, 0]
    elif layer_type == 'Embedding':
        return [0, 1, 0, 0]
    elif layer_type == 'Linear':
        return [0, 0, 1, 0]
    elif layer_type == 'Dropout':
        return [0, 0, 0, 1]
    else:
        print(layer_type)
        raise ValueError("Unknown layer type")


def encode_layer_cnn(layer_type):
    if layer_type in ['adaptive_avg_pool2d', 'Sigmoid', 'softmax', 'ELU', 'GELU', 'Identity', 'LeakyReLU', 'Mish', 'PReLU', 'ReLU', 'SELU', 'SiLU', 'Softplus', 'Tanh']:
        return [1, 0]
    elif layer_type == 'conv2d':
        return [0, 1]
    elif layer_type == 'linear':
        return [1, 1]
    elif layer_type in ['dropout','batchnorm2d']:
        return [0, 0]
    else:
        print(layer_type)
        raise ValueError("Unknown layer type")

def encode_layer_mlp(layer_type):
    if layer_type in ['dropout','batch_normalization', 'Sigmoid', 'Softmax', 'ELU', 'GELU', 'Identity', 'LeakyReLU', 'Mish', 'PReLU', 'ReLU', 'SELU', 'SiLU', 'Softplus', 'Tanh']:
        return [1, 0]
    elif layer_type == 'linear':
        return [0, 1]
    else:
        print(layer_type)
        raise ValueError("Unknown layer type")


def process_sequence_data(sequence, data_type="transformer"):
    processed_sequence = []
    for entry in eval(sequence):  # Evaluate string as list of tuples
        layer_type, feature_1, feature_2 = entry
        if data_type=="transformer":
            encoded_layer = encode_layer_transformer(layer_type)
        elif data_type=="cnn":
            encoded_layer = encode_layer_cnn(layer_type)
        elif data_type=="mlp":
            encoded_layer = encode_layer_mlp(layer_type)

        combined = encoded_layer + [feature_1, feature_2]
        processed_sequence.append(combined)
    return np.array(processed_sequence)


def get_max_layer(df, column_name='Activations-Params'):
    max_layer=0
    # Check if the column 'params_activations_list' exists
    if column_name in df.columns:
        for i in range(len(df)):
            try:
                # Check if the entry is already a list; if not, use ast.literal_eval to convert it
                entry = df[column_name].iloc[i]  # Use iloc to access the row by position
                if isinstance(entry, str):
                    current_list = ast.literal_eval(entry)
                else:
                    current_list = entry
                # Ensure the parsed content is a list
                if isinstance(current_list, list):
                    l = len(current_list)
                    if l > max_layer:
                        max_layer = l
                else:
                    print(f"Unexpected format at index {i}: {entry}")

            except (ValueError, SyntaxError) as e:
                print(f"Error processing entry at index {i}: {entry} - {e}")
        print('Maximum layers:', max_layer)
        return max_layer
    else:
        print("Column 'params_activations_list' does not exist in the DataFrame.")

def get_filtered_lists(list1, list2, list3):
    filtered_list1, filtered_list2, filtered_list3 = zip(*[(l1, l2, l3) for l1, l2, l3 in zip(list1, list2, list3) if len(l1) > 0])
    return filtered_list1, filtered_list2, filtered_list3

def transformer_data4transformer(config, kfold=False):
    csv_file_path = os.path.join(config["data_dir"], "trans_data.csv")
    df = pd.read_csv(csv_file_path)

    df.loc[df['Status'] == 'OOM_CRASH', 'Max GPU Memory (MiB)'] = 45000

    df = df[['Depth', 'Batch Size', 'Activations-Params', 'Total Activations', 'Total Parameters',
        'Max GPU Memory (MiB)', 'NonDynamicallyQuantizableLinear Count',
        'Linear Count', 'LayerNorm Count', 'Dropout Count']]
    df['Total_Activations_Batch_Size'] = df['Total Activations'] * df['Batch Size']

    df = df.reset_index(drop=True)

    max_seq_len = get_max_layer(df)

    bins = [0, 8000, 16000, 24000, 32000, 40000, 50000]
    labels = [i for i in range(0, len(bins) - 1)]

    df['memory_usage_label'] = pd.cut(df['Max GPU Memory (MiB)'], bins=bins, labels=labels, right=False)

    # desired_features = df[['Activations-Params','Depth', 'Batch Size', 'Total Activations', 'Total Parameters',
    #         'NonDynamicallyQuantizableLinear Count', 'Linear Count', 'LayerNorm Count', 'Dropout Count']].values
    desired_features = df[['Activations-Params', 'Depth','Total Activations', 'Total Parameters', 'Batch Size',
                      'NonDynamicallyQuantizableLinear Count', 'Linear Count', 'Total_Activations_Batch_Size',
                      'LayerNorm Count', 'Dropout Count']].values
    desired_labels = df['memory_usage_label'].values
    class_counts = len(df['memory_usage_label'].value_counts())
    data_related_info = {"class_counts":class_counts, "max_seq_len":max_seq_len, 
                         "extra_features_num":desired_features.shape[1], }
    if not kfold:
        x_train, x_temp, y_train, y_temp = train_test_split(
            desired_features, desired_labels, test_size=0.3, random_state=42)

        x_test, x_val, y_test, y_val = train_test_split(
            x_temp, y_temp, test_size=0.66, random_state=42)

        train_dataset = LayerSequenceDataset(x_train, y_train, max_seq_len)
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

        val_dataset = LayerSequenceDataset(x_val, y_val, max_seq_len)
        val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        test_dataset = LayerSequenceDataset(x_test, y_test, max_seq_len)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader, data_related_info
    else:
        whole_dataset = LayerSequenceDataset(desired_features, desired_labels, max_seq_len)
        return whole_dataset, data_related_info
    
def cnn_data4transformer(config, kfold=False):
    csv_file_path = os.path.join(config["data_dir"], "cnn_data1.csv")
    df = pd.read_csv(csv_file_path)
    df['Total_Activations_Batch_Size'] = df['Total Activations'] * df['Batch Size']
    df.loc[df['Status'] == 'OOM_CRASH', 'Max GPU Memory (MiB)'] = 45000

    df = df.dropna(subset=['Activation Function'])

    # print(df['Activation Function'])
    df = df[df['Activation Function'] != 'None']
    print(df['Activation Function'].unique())

    df = df[~df['architecture'].isin(['residual', 'dense'])]

    df = df[['Max GPU Memory (MiB)', 'Depth', 'Batch Size', 'Total Parameters', 'Total Activations', 'Activations-Params', 'Activation Function',
            'Total_Activations_Batch_Size','Conv2d Count', 'BatchNorm2d Count', 'Dropout Count',
            'Input Size (MB)', 'Forward/Backward Pass Size (MB)', 'Params Size (MB)',
            'Estimated Total Size (MB)']]
    
    df = df.rename(columns={
        'Max GPU Memory (MiB)': 'real_memory_usage',
        'Depth': 'layers',
        'Batch Size': 'batch_size',
        'Total Parameters': 'all_parameters',
        'Total Activations': 'all_activations',
        'Activations-Params': 'params_activations_list'
    })
    max_seq_len = get_max_layer(df, column_name='params_activations_list')
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
    df['memory_usage_label'] = pd.cut(df['real_memory_usage'], bins=bins, labels=labels, right=False)

    desired_features = df[['params_activations_list', 'batch_size', 'all_parameters', 'all_activations', 'Total_Activations_Batch_Size',
                        'Conv2d Count', 'BatchNorm2d Count','Dropout Count',
                        # 'Input Size (MB)', 'Forward/Backward Pass Size (MB)', 'Params Size (MB)', 'Estimated Total Size (MB)',
                        'activation_encoding_sin', 'activation_encoding_cos',
                    ]].values
    desired_labels = df['memory_usage_label'].values
    class_counts = len(df['memory_usage_label'].value_counts())

    data_related_info = {"class_counts":class_counts, "max_seq_len":max_seq_len, 
                         "extra_features_num":desired_features.shape[1], }
    if not kfold:
        x_train, x_temp, y_train, y_temp = train_test_split(
            desired_features, desired_labels, test_size=0.3, random_state=42)

        x_test, x_val, y_test, y_val = train_test_split(
            x_temp, y_temp, test_size=0.66, random_state=42)

        train_dataset = LayerSequenceDataset(x_train, y_train, max_seq_len, data_type='cnn')
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

        val_dataset = LayerSequenceDataset(x_val, y_val, max_seq_len, data_type='cnn')
        val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        test_dataset = LayerSequenceDataset(x_test, y_test, max_seq_len, data_type='cnn')
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader, data_related_info
    else:
        whole_dataset = LayerSequenceDataset(desired_features, desired_labels, max_seq_len, data_type='cnn')
        return whole_dataset, data_related_info


def mlp_data4transformer(config, kfold=False):
    csv_file_path = os.path.join(config["data_dir"], "mlp_data2.csv")
    df = pd.read_csv(csv_file_path)
    df = df[['Max GPU Memory (MiB)', 'Depth', 'Batch Size', 'Total Parameters', 'Total Activations', 'Activations-Params']]

    df['Total_Activations_Batch_Size'] = df['Total Activations'] * df['Batch Size']

    # Map the existing columns to your desired column structure
    df = df.rename(columns={
        'Max GPU Memory (MiB)': 'real_memory_usage',
        'Depth': 'layers',
        'Batch Size': 'batch_size',
        'Total Parameters': 'all_parameters',
        'Total Activations': 'all_activations',
        'Activations-Params': 'params_neurons_list'
    })
        
    max_seq_len = get_max_layer(df, column_name='params_neurons_list')
    bins = [i*1000 for i in range(1, 6)]  # Define your bin edges
    bins.append(float('inf'))
    labels = [i for i in range(len(bins)-1)]

    df['memory_usage_label'] = pd.cut(df['real_memory_usage'], bins=bins, labels=labels, right=False)

    desired_features = df[['params_neurons_list', 'batch_size', 'all_parameters', 'all_activations', 'Total_Activations_Batch_Size']].values
    desired_labels = df['memory_usage_label'].values
    class_counts = len(df['memory_usage_label'].value_counts())

    data_related_info = {"class_counts":class_counts, "max_seq_len":max_seq_len, 
                         "extra_features_num":desired_features.shape[1], }
    if not kfold:
        x_train, x_temp, y_train, y_temp = train_test_split(
            desired_features, desired_labels, test_size=0.3, random_state=42)

        x_test, x_val, y_test, y_val = train_test_split(
            x_temp, y_temp, test_size=0.66, random_state=42)

        train_dataset = LayerSequenceDataset(x_train, y_train, max_seq_len, data_type='mlp')
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

        val_dataset = LayerSequenceDataset(x_val, y_val, max_seq_len, data_type='mlp')
        val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        test_dataset = LayerSequenceDataset(x_test, y_test, max_seq_len, data_type='mlp')
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader, data_related_info
    else:
        whole_dataset = LayerSequenceDataset(desired_features, desired_labels, max_seq_len, data_type='mlp')
        return whole_dataset, data_related_info