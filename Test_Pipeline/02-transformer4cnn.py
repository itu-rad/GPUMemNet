import os
import pickle
import re
import numpy as np
from collections import Counter

import torch 
from single_transformer_model import classification_gpu_usage

# activation function preparation
# List of activation functions and their positional encodings
print("the activation function are positionally encoded safely")
activations = ['ELU', 'GELU', 'LeakyReLU', 'Mish', 'PReLU', 'ReLU', 'SELU', 'SiLU', 'Softplus', 'Tanh']
max_seq_len=119

# Positional encoding function (already defined in your code)
def positional_encoding_2d(num_states):
    positions = []
    for i in range(num_states):
        position = (np.sin(i * np.pi / num_states), np.cos(i * np.pi / num_states))
        positions.append(position)
    return np.array(positions)

# Generate positional encodings
positional_encodings = positional_encoding_2d(len(activations))
activation_to_encoding = {activation: positional_encodings[i] for i, activation in enumerate(activations)}

# Function to get the encoding for a given activation function
def get_activation_encoding(activation):

    if activation == "ReLU6":
        activation = "ReLU"

    return activation_to_encoding[activation]


def most_frequent_activation_function(activation_functions):
    # Count the occurrences of each activation function
    activation_counter = Counter(activation_functions)
    
    # Find the activation function with the highest count
    # most_common_activation, count = activation_counter.most_common(1)[0]
    most_common_activation, _ = activation_counter.most_common(1)[0]
    
    return most_common_activation



def extract_model_info(out_file, batch_size):
    try:
        with open(out_file, 'r') as file:
            lines = file.readlines()

        if not lines:  # If the file is empty
            return [], 'None', 0, 0, 0, 0.0, 0.0, 0.0, 0.0, {
                'conv2d': 0,
                'batchnorm2d': 0,
                'dropout': 0,
                'adaptive_avg_pool2d': 0,
                'linear': 0,
                'softmax': 0
            }

        activations_params = []
        activation_functions_list = []
        total_params = 0
        total_activations = 0
        depth = 0  # Counting Conv2d layers

        # Initialize memory size variables
        input_size_mb = 0.0
        forward_backward_size_mb = 0.0
        params_size_mb = 0.0
        estimated_total_size_mb = 0.0

        # Dictionary to keep track of the count of each layer type
        layer_counts = {
            'conv2d': 0,
            'batchnorm2d': 0,
            'dropout': 0,
            'adaptive_avg_pool2d': 0,
            'linear': 0,
            'softmax': 0
        }

        temp = 0
        for line in lines:
            # Skip 'Block' lines
            if "SqueezeExcitation" in line:
                continue

            if "MBConv" in line:
                continue

            if "Block" in line:
                continue
            
            if "SeparableConv2d" in line:
                continue

            if "InvertedResidual" in line:
                continue

            if "BasicConv2d" in line:
                continue


            # Handle Conv2d layers
            if "Conv2d-" in line:
                depth += 1
                layer_counts['conv2d'] += 1
                output_shape = re.findall(r'\[\-?\d*, (\d+), (\d+), (\d+)\]', line)
                param_count = re.findall(r'(\d{1,3}(?:,\d{3})*)$', line)
                if output_shape and param_count:
                    channels = int(output_shape[0][0])
                    height = int(output_shape[0][1])
                    width = int(output_shape[0][2])
                    activations = channels * height * width
                    params = int(param_count[0].replace(',', ''))

                    activations_params.append(('conv2d', activations * batch_size, params))
                    total_params += params
                    total_activations += activations
                else:
                    print(f"Warning: Missing Conv2d data in {out_file}, line: {line.strip()}")

            # Handle BatchNorm2d layers
            elif "BatchNorm2d-" in line:
                layer_counts['batchnorm2d'] += 1
                output_shape = re.findall(r'\[\-?\d*, (\d+), (\d+), (\d+)\]', line)
                param_count = re.findall(r'(\d{1,3}(?:,\d{3})*)$', line)

                if output_shape and param_count:
                    channels = int(output_shape[0][0])
                    height = int(output_shape[0][1])
                    width = int(output_shape[0][2])
                    activations = channels * height * width
                    params = int(param_count[0].replace(',', ''))

                    activations_params.append(('batchnorm2d', activations * batch_size, params))
                    total_params += params
                    total_activations += activations
                else:
                    print(f"Warning: Missing BatchNorm2d data in {out_file}, line: {line.strip()}")

            # Handle Dropout layers (4D and 2D)
            elif "Dropout-" in line:
                layer_counts['dropout'] += 1
                output_shape_4d = re.findall(r'\[\-?\d*, (\d+), (\d+), (\d+)\]', line)
                output_shape_2d = re.findall(r'\[\-?\d*, (\d+)\]', line)
                
                if output_shape_4d:
                    channels = int(output_shape_4d[0][0])
                    height = int(output_shape_4d[0][1])
                    width = int(output_shape_4d[0][2])
                    activations = channels * height * width
                    activations_params.append(('dropout', activations * batch_size, 0))
                    total_activations += activations

                elif output_shape_2d:
                    activations = int(output_shape_2d[0][0])
                    activations_params.append(('dropout', activations * batch_size, 0))
                    total_activations += activations

                else:
                    print(f"Warning: Missing Dropout data in {out_file}, line: {line.strip()}")

            # Handle AdaptiveAvgPool2d layers
            elif "AdaptiveAvgPool2d-" in line:
                layer_counts['adaptive_avg_pool2d'] += 1
                output_shape = re.findall(r'\[\-?\d*, (\d+), (\d+), (\d+)\]', line)
                if not output_shape:
                    output_shape = re.findall(r'\[\-?\d*, (\d+), (\d+)\]', line)  # Fallback for shapes like [C, 1, 1]
                if not output_shape:
                    output_shape = re.findall(r'\[\-?\d*, (\d+)\]', line)  # Fallback for shapes like [C]
                if output_shape:
                    channels = int(output_shape[0][0])
                    activations = channels  # AdaptiveAvgPool2d typically reduces spatial dimensions to 1x1
                    activations_params.append(('adaptive_avg_pool2d', activations * batch_size, 0))
                    total_activations += activations
                else:
                    print(f"Warning: Missing AdaptiveAvgPool2d data in {out_file}, line: {line.strip()}")

            # Handle Linear layers
            elif "Linear-" in line:
                layer_counts['linear'] += 1
                output_shape = re.findall(r'\[\-?\d*, (\d+)\]', line)
                param_count = re.findall(r'(\d{1,3}(?:,\d{3})*)$', line)
                if output_shape and param_count:
                    activations = int(output_shape[0])
                    params = int(param_count[0].replace(',', ''))
                    activations_params.append(('linear', activations * batch_size, params))
                    total_params += params
                    total_activations += activations
                else:
                    print(f"Warning: Missing Linear data in {out_file}, line: {line.strip()}")

            # Handle Softmax layers
            elif "Softmax-" in line:
                layer_counts['softmax'] += 1
                output_shape = re.findall(r'\[\-?\d*, (\d+)\]', line)  # Softmax typically has a 1D output
                if output_shape:
                    activations = int(output_shape[0])
                    activations_params.append(('softmax', activations * batch_size, 0))
                    total_activations += activations
                else:
                    print(f"Warning: Missing Softmax data in {out_file}, line: {line.strip()}")

            # Handle Activation Functions (e.g., ReLU, LeakyReLU) with both 2D and 4D output shapes
            elif re.search(r'(ReLU6|ReLU|LeakyReLU|PReLU|ELU|SELU|GELU|Tanh|SiLU|Softplus|Mish)-\d+', line):
                activation_func = re.findall(r'(ReLU6|ReLU|LeakyReLU|PReLU|ELU|SELU|GELU|Tanh|SiLU|Softplus|Mish)-\d+', line)[0]
                
                # Try to match both 4D and 2D output shapes
                output_shape_4d = re.findall(r'\[\-?\d*, (\d+), (\d+), (\d+)\]', line)
                output_shape_2d = re.findall(r'\[\-?\d*, (\d+)\]', line)
                
                if output_shape_4d:
                    channels = int(output_shape_4d[0][0])
                    height = int(output_shape_4d[0][1])
                    width = int(output_shape_4d[0][2])
                    activations = channels * height * width
                    activations_params.append((activation_func, activations * batch_size, 0))  # No parameters for activation functions
                    total_activations += activations
                    activation_functions_list.append(activation_func)

                elif output_shape_2d:
                    activations = int(output_shape_2d[0][0])
                    activations_params.append((activation_func, activations * batch_size, 0))  # No parameters for activation functions
                    total_activations += activations
                    activation_functions_list.append(activation_func)

                else:
                    print(f"Warning: Missing activation data for {activation_func} in {out_file}, line: {line.strip()}")

            # Handling Total params:
            if "Total params:" in line:
                temp = re.findall(r'Total params:\s*([\d,]+)', line)
                if temp:
                    temp = int(temp[0].replace(',', ''))

            # Handling input and memory sizes
            if "Input size (MB):" in line:
                input_size_mb = float(re.findall(r'Input size \(MB\):\s*([\d.]+)', line)[0])
            elif "Forward/backward pass size (MB):" in line:
                forward_backward_size_mb = float(re.findall(r'Forward/backward pass size \(MB\):\s*([\d.]+)', line)[0])
            elif "Params size (MB):" in line:
                params_size_mb = float(re.findall(r'Params size \(MB\):\s*([\d.]+)', line)[0])
            elif "Estimated Total Size (MB):" in line:
                estimated_total_size_mb = float(re.findall(r'Estimated Total Size \(MB\):\s*([\d.]+)', line)[0])

        # Final consistency check
        if temp != total_params:
            print(temp, total_params)
            # exit()
            # raise ValueError("Mismatch between total params in summary and parsed params.")


        if len(activation_functions_list) > 0:
            activation_function = most_frequent_activation_function(activation_functions_list)
        else:
            activation_function = "ReLU"

        return activations_params, activation_function, depth, total_params, total_activations, input_size_mb, forward_backward_size_mb, params_size_mb, estimated_total_size_mb, layer_counts

    except Exception as e:
        print(f"Error processing {out_file}: {e}")
        return [], 'None', 0, 0, 0, 0.0, 0.0, 0.0, 0.0, {
            'conv2d': 0,
            'batchnorm2d': 0,
            'dropout': 0,
            'adaptive_avg_pool2d': 0,
            'linear': 0,
            'softmax': 0
        }


def encode_layer_cnn(layer_type):
    if layer_type in ['adaptive_avg_pool2d', 'Sigmoid', 'softmax', 'ELU', 'GELU', 'Identity', 'LeakyReLU', 'Mish', 'PReLU', 'ReLU', 'SELU', 'SiLU', 'Softplus', 'Tanh', "ReLU6"]:
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

# Function to process each sequence
def process_sequence(sequence):
    processed_sequence = []
    for entry in sequence:  # Evaluate string as list of tuples
        layer_type, feature_1, feature_2 = entry
        # for fc
        # encoded_layer = encode_layer(layer_type)
        # for cnn
        encoded_layer = encode_layer_cnn(layer_type)
        combined = encoded_layer + [feature_1, feature_2]
        processed_sequence.append(combined)
    return np.array(processed_sequence)

def get_filtered_lists(list1, list2):
    filtered_list1, filtered_list2 = zip(*[(l1, l2) for l1, l2 in zip(list1, list2) if len(l1) > 0])
    return filtered_list1, filtered_list2


def LayerSequenceDataset(seq, max_seq_len):
    x_data = process_sequence(seq[0])
    additional_features = np.array(seq[1:]).astype(np.int64)
    x_data, _ = get_filtered_lists(x_data, additional_features)
    sequence = x_data
    # Padding sequences to max length
    if len(sequence) < max_seq_len:
        padded_sequence = np.pad(sequence, ((0, max_seq_len - len(sequence)), (0,0)), 'constant')
    else:
        padded_sequence = sequence[:max_seq_len]
    return torch.tensor(padded_sequence, dtype=torch.float32).unsqueeze(0), torch.tensor(additional_features, dtype=torch.float32).unsqueeze(0)


# def LayerSequenceDataset(seq, max_seq_len):
#     self.x_data = [process_sequence(seq[0]) for seq in x_data]
#     self.batch_size = [np.array(seq[1:]).astype(np.int64) for seq in x_data]
#     self.y_labels = y_labels
#     self.x_data, self.batch_size, self.y_labels = get_filtered_lists(self.x_data, self.batch_size, self.y_labels)
#     self.max_seq_len = max_seq_len


#     sequence = self.x_data[idx]
#     batch_size = self.batch_size[idx]
#     label = self.y_labels[idx]
#     # Padding sequences to max length
#     if len(sequence) < self.max_seq_len:
#         padded_sequence = np.pad(sequence, ((0, self.max_seq_len - len(sequence)), (0,0)), 'constant')
#     else:
#         padded_sequence = sequence[:self.max_seq_len]

#     return torch.tensor(padded_sequence, dtype=torch.float32), torch.tensor(batch_size, dtype=torch.float32)



def process_model_files(directory, model_file):
    # Load the pre-trained model from the pickled file
    # Load the model from the .ckpt file
    checkpoint_path = model_file
    output_size = 6  # Set the output size according to your task


    # Load the model with its weights from the checkpoint
    model = classification_gpu_usage.load_from_checkpoint(checkpoint_path, num_features=4, num_classes=6, d_model=6, nhead=3,
                                           num_layers=3, dim_feedforward=8, dropout=0, max_seq_len=max_seq_len)


    # Print the state_dict to see if the weights are loaded (you can print keys or size)
    # state_dict = model.state_dict()
    
    # # Checking some key parameters, for example, the first few layers
    # for param_tensor in list(state_dict.keys())[:10]:  # Display first 10 layers for brevity
    #     print(f"Layer: {param_tensor}, Size: {state_dict[param_tensor].size()}")
    
    
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.model'):
            # Extract batch size from the filename (e.g., "modelName_batchSize.model")
            match = re.search(r'_(\d+)\.model$', filename)
            if not match:
                print(f"Batch size not found in file name: {filename}")
                continue
            
            batch_size = int(match.group(1))
            
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            
            # Extract model information using the parser function
            features = extract_model_info(file_path, batch_size)
            
            model.eval()

            input_features = prepare_features_for_model(features, batch_size)
            x, b = LayerSequenceDataset(input_features, max_seq_len)

            with torch.no_grad():
                logits = model(x, b)
                predictions = torch.argmax(logits, dim=1)  # Assuming classification task
            print(filename, "Predictions:", predictions)
            


            activations = input_features[0][0]
            parameters = input_features[0][1]
            batch_size = input_features[0][2]
            gradients = parameters

            # horus_formula_estimation = (activations * batch_size + parameters) + (batch_size * gradients)

            # horus_in_bytes = horus_formula_estimation * 4

            # horus_estimations_MB = horus_in_bytes / (1024 ** 2)

            # print("Horus Formual Estimation: ", horus_estimations_MB, activations, parameters, batch_size)

            

def prepare_features_for_model(features, batch_size):
    """
    Convert the features extracted by extract_model_info into the format
    required by the loaded model.
    
    Modify this function to extract relevant features.
    """
    
    
    activations_params, activation_function, depth, total_params, total_activations, input_size_mb, forward_backward_size_mb, params_size_mb, estimated_total_size_mb, layer_counts = features
    
    activation_encoding = get_activation_encoding(activation_function)

    if not 'conv2d' in layer_counts.keys():
        layer_counts['conv2d'] = 0
    if not 'batchnorm2d' in layer_counts.keys():
        layer_counts['batchnorm2d'] = 0
    if not 'dropout' in layer_counts.keys():
        layer_counts['dropout'] = 0

    # Packing the features into a list for mlp cnn estimator (modify as needed)
    feature_list_mlp = [
        activations_params, 
        batch_size,                         # Feature 3
        total_params,                       # Feature 2
        total_activations,                  # Feature 1
        total_activations * batch_size,     # Feature 4
        layer_counts['conv2d'],             # Feature 5
        layer_counts['batchnorm2d'],        # Feature 6
        layer_counts['dropout'],            # Feature 7
        activation_encoding[0],             # Feature 8 (use positional encoding for the activation function)
        activation_encoding[1]              # Feature 9 (use positional encoding for the activation function)
    ]
    

    return feature_list_mlp

# Usage example
if __name__ == "__main__":
    directory = 'cnn_models'  # Specify the directory containing .model files
    # model_file = 'estimator/cnn_mlp_8g.ckpt'  # Specify the path to the pickled model
    
    model_file = '../Analysis/00-Cleaned-NoteBooks/003-Transformer-based-estimators/transformer4cnn.ckpt'  # Specify the path to the pickled model

    process_model_files(directory, model_file)
