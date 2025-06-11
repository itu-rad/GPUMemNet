import os
import sys
import re
import numpy as np
from collections import Counter
from collections import defaultdict

import torch

# Get the parent directory (i.e., one level up)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Ensemble"))
sys.path.append(PROJECT_ROOT)


from utils import read_yaml
from models.transformer_models import TransformerEnsemble
from Tester_trans_mlp import extract_model_info, process_model_files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_seq_len=314

# Configuration loading
def _get_config():
    return read_yaml("../Ensemble/config.yaml")


def infer_transformer4cnn(x, batch_size_feature):
    transformers_architectures = read_yaml("../Ensemble/transformers_architectures.yaml")['model_configs']
    config = _get_config()
    model = TransformerEnsemble.load_from_checkpoint(
        "../Ensemble/trained_models/transformer4transformer/transformer4transformer.ckpt",
        model_configs = transformers_architectures,
        num_features=6, 
        num_classes=6, 
        learning_rate=config["learning_rate"], 
        max_seq_len=max_seq_len, 
        extra_fetures_num=9,
    )

    model.to(device)

    # state_dict = model.state_dict()

    #    # Checking some key parameters, for example, the first few layers
    # for param_tensor in list(state_dict.keys())[:10]:  # Display first 10 layers for brevity
    #     print(f"Layer: {param_tensor}, Size: {state_dict[param_tensor].size()}")


    model.eval()

    with torch.no_grad():
        x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        batch_size_feature = torch.tensor(batch_size_feature, dtype=torch.float32, device=device).unsqueeze(0)
        # print(x.shape, batch_size_feature.shape)
        logits = model(x, batch_size_feature)
        return logits.argmax(dim=1).item()
    

# activation function preparation
# List of activation functions and their positional encodings
print("the activation function are positionally encoded safely")
activations = ['ELU', 'GELU', 'LeakyReLU', 'Mish', 'PReLU', 'ReLU', 'SELU', 'SiLU', 'Softplus', 'Tanh']

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



def get_batchsize_and_sequrenceLength(input_string):
    batch_size_match = re.search(r'bs:(\d+)', input_string)
    sequence_length_match = re.search(r'sl:(\d+)', input_string)

    batch_size = int(batch_size_match.group(1)) if batch_size_match else None
    sequence_length = int(sequence_length_match.group(1)) if sequence_length_match else None

    return batch_size, sequence_length


def extract_model_info(file_path, batch_size, sequence_length):
    with open(file_path, 'r') as file:
            lines = file.readlines()


    # if not summary.strip():  # Check if the summary is empty
    #     return None

    # lines = summary.split('\n')

    activations_params = []
    total_params = 0
    total_activations = 0
    accumulated_params = 0  # Accumulated parameters over transformer layers
    depth = 0  # Count of layers

    # Dictionary to keep track of the count of each layer type
    layer_counts = defaultdict(int)

    # Initial number of activations is based on the sequence length and batch size
    current_activations = sequence_length

    # Regex patterns for extracting layers and parameters
    embedding_pattern = r'Embedding\((\d+), (\d+)\),\s*([\d,]+)\s*params'
    linear_pattern = r'Linear\(.*?\),\s*([\d,]+)\s*params'
    non_dynamically_linear_pattern = r'NonDynamicallyQuantizableLinear\(.*?\),\s*([\d,]+)\s*params'
    activations_pattern = r'in_features=(\d+),\s*out_features=(\d+)'
    layernorm_pattern = r'LayerNorm\(.*?\),\s*([\d,]+)\s*params'
    conv1d_pattern = r'Conv1D\(.*?\),\s*([\d,]+)\s*params'


    for line in lines:

        total_params_match = re.search(r'Total params:\s*([\d,]+)', line)
        if total_params_match:
            total_params_str = total_params_match.group(1).replace(',', '')  # Remove commas
            total_params = int(total_params_str)

        # First check for Embedding layer
        embedding_match = re.search(embedding_pattern, line)
        if embedding_match:
            vocab_size, embedding_dim, params = embedding_match.groups()
            params = int(params.replace(',', ''))  # Remove commas and convert to int

            # Calculate the number of activations for the embedding layer
            current_activations = sequence_length * int(embedding_dim)

            # Append the layer and its activations to the list
            activations_params.append(('Embedding', current_activations * batch_size , params))
            total_activations += current_activations

            # Increment layer count
            layer_counts['Embedding'] += 1
            depth += 1

            # Skip further checks for this line since it's already processed
            continue

        # Now check for NonDynamicallyQuantizableLinear
        non_dyn_linear_match = re.search(non_dynamically_linear_pattern, line)
        if non_dyn_linear_match:
            params = int(non_dyn_linear_match.group(1).replace(',', ''))

            # Extract in_features and out_features to update current_activations for NonDynamicallyQuantizableLinear layers
            activation_match = re.search(activations_pattern, line)
            if activation_match:
                in_features, out_features = map(int, activation_match.groups())
                current_activations = out_features   # Update the activations

            # Append the layer and its activations to the list
            activations_params.append(('NonDynamicallyQuantizableLinear', current_activations * batch_size, params))
            total_activations += current_activations

            # Increment layer count
            layer_counts['NonDynamicallyQuantizableLinear'] += 1
            depth += 1

            # Skip further checks for this line since it's already processed
            continue

        # Now check for Linear layer information
        linear_match = re.search(linear_pattern, line)
        if linear_match:
            params = int(linear_match.group(1).replace(',', ''))

            # Extract in_features and out_features to update current_activations for Linear layers
            activation_match = re.search(activations_pattern, line)
            if activation_match:
                in_features, out_features = map(int, activation_match.groups())
                current_activations = out_features   # Update the activations

            # Append the layer and its activations to the list
            activations_params.append(('Linear', current_activations * batch_size, params))
            total_activations += current_activations

            # Increment layer count
            layer_counts['Linear'] += 1
            depth += 1

        # Handle LayerNorm and Dropout layers, which do not change activations
        elif 'LayerNorm' in line:
            layernorm_match = re.search(layernorm_pattern, line)

            if layernorm_match:
                params = int(layernorm_match.group(1).replace(',', ''))  # Extract params directly
            else:
                print("there is a problem")
                exit()
                params = 0  # Default to zero if not found

            
            layer_counts['LayerNorm'] += 1
            activations_params.append(('LayerNorm', current_activations, params))  # No params for LayerNorm
            total_activations += current_activations
            depth += 1

        elif 'Dropout' in line:
            layer_counts['Dropout'] += 1
            activations_params.append(('Dropout', current_activations, 0))  # No params for Dropout
            total_activations += current_activations
            depth += 1
        
        elif 'Conv1D' in line:
            conv1d_match = re.search(conv1d_pattern, line)
            if conv1d_match:
                params = int(conv1d_match.group(1).replace(',', ''))

                in_features = current_activations
                # out_features = (params // (in_features + 1))  # Solve for out_features

                out_features = current_activations

                # current_activations = sequence_length * out_features

                # Append the layer and its activations to the list
                # activations_params.append(('Conv1D', current_activations * batch_size, params))
                activations_params.append(('Linear', current_activations * batch_size, params))

                total_activations += current_activations

                # Increment layer count
                layer_counts['Conv1D'] += 1
                
                # in this version, our model is unaware of conv layers, so we give them as linear layers
                layer_counts['Linear'] += 1

                depth += 1

            elif 'NewGELUActivation' in line:
                # Since there are no parameters, we set params to 0
                params = 0
                activations_params.append(('NewGELUActivation', current_activations * batch_size, params))
                total_activations += current_activations

                # Increment layer count
                # layer_counts['NewGELUActivation'] += 1
                depth += 1

    return {
        "activations_params": activations_params,
        "total_params": total_params,
        "total_activations": total_activations,
        "accumulated_params": accumulated_params,
        "layer_counts": dict(layer_counts),
        "depth": depth
    }



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
    

def process_sequence_data(sequence, data_type):
    processed_sequence = []
    for entry in sequence:  # Evaluate string as list of tuples
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


def get_filtered_lists(list1, list2):
    filtered_list1, filtered_list2 = zip(*[(l1, l2) for l1, l2 in zip(list1, list2) if len(l1) > 0])
    return filtered_list1, filtered_list2


def LayerSequenceDataset(seq, max_seq_len, data_type="transformer"):
    x_data = process_sequence_data(seq[0], data_type=data_type)
    additional_features = np.array(seq[1:]).astype(np.int64)
    x_data, additional_features = get_filtered_lists(x_data, additional_features)
    sequence = x_data
    # Padding sequences to max length
    if len(sequence) < max_seq_len:
        padded_sequence = np.pad(sequence, ((0, max_seq_len - len(sequence)), (0,0)), 'constant')
    else:
        padded_sequence = sequence[:max_seq_len]
    return torch.tensor(padded_sequence, dtype=torch.float32), torch.tensor(additional_features, dtype=torch.float32)


def process_model_files(directory):


    
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # if "gpt" in filename:
        #     continue
        
        if filename.endswith('.txt'):
            # Extract batch size from the filename (e.g., "modelName_batchSize.model")
            
            batch_size, sequence_length = get_batchsize_and_sequrenceLength(filename)
            
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            
            # Extract model information using the parser function
            features = extract_model_info(file_path, batch_size, sequence_length)
            
            

            _, input_features = prepare_features_for_model(features, batch_size)
            padded_sequence, additional_features = LayerSequenceDataset(input_features, max_seq_len=max_seq_len)
            # print(file_path, "\n", input_features)
            
            # input_features = torch.tensor(input_features, dtype=torch.float32)
            # input_features = input_features.view(1, -1)  # Reshape if necessary

            predictions = infer_transformer4cnn(padded_sequence, additional_features)

            print(filename, "Predictions:", predictions)


            activations = input_features[0][1]
            parameters = input_features[0][3]
            batch_size = input_features[0][4]
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
    
    # Unpack the features (adapt depending on what features your model needs)

    activations_params = features["activations_params"]
    total_params = features["total_params"], 
    total_activations = features["total_activations"] 
    accumulated_params = features["accumulated_params"], 
    layer_counts = features["layer_counts"]
    depth = features["depth"]
    

    # the order, we should pass the data to the model
    # 'Depth','Total Activations', 'Total_Activations_Batch_Size', 'Total Parameters', 'Batch Size',
    #                       'Linear Count', 'LayerNorm Count', 'Dropout Count'
    
    feature_list_mlp = [
        depth,                                   # Feature 1
        total_activations,                       # Feature 2
        total_activations * batch_size,          # Feature 3
        total_params[0],                            # Feature 4
        batch_size,                              # Feature 5
        layer_counts['Linear'],                  # Feature 6
        layer_counts['LayerNorm'],               # Feature 7
        layer_counts['Dropout'],                 # Feature 8
    ]

    if not "NonDynamicallyQuantizableLinear" in layer_counts.keys():
        layer_counts['NonDynamicallyQuantizableLinear'] = 0
    # Packing the features into a list for transformer-based cnn estimator (modify as needed)
    feature_list_transformer = [
        activations_params,
        depth,                                   # Feature 1
        total_activations,                       # Feature 2
        total_params[0],                            # Feature 4
        batch_size,                              # Feature 5
        layer_counts['NonDynamicallyQuantizableLinear'],
        layer_counts['Linear'],                  # Feature 6
        total_activations * batch_size,          # Feature 3
        layer_counts['LayerNorm'],               # Feature 7
        layer_counts['Dropout'],                 # Feature 8
    ]
    
    # Add other features as necessary for your model
    return feature_list_mlp, feature_list_transformer

# Usage example
if __name__ == "__main__":
    directory = 'Trans_models'  # Specify the directory containing .model files
    process_model_files(directory)