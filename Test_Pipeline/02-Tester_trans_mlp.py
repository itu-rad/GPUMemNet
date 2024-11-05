import os
import pickle
import re
import numpy as np
from collections import Counter

import torch 

from Trans_MLPredictor import classification_gpu_usage

from collections import defaultdict


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



def process_model_files(directory, model_file):
    # Load the pre-trained model from the pickled file
    # Load the model from the .ckpt file
    checkpoint_path = model_file
    output_size = 6  # Set the output size according to your task


    # Load the model with its weights from the checkpoint
    model = classification_gpu_usage.load_from_checkpoint(checkpoint_path, output_size=output_size)


    # # Print the state_dict to see if the weights are loaded (you can print keys or size)
    state_dict = model.state_dict()
    
    # Checking some key parameters, for example, the first few layers
    for param_tensor in list(state_dict.keys())[:10]:  # Display first 10 layers for brevity
        print(f"Layer: {param_tensor}, Size: {state_dict[param_tensor].size()}")
    
    
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
            
            
            model.eval()

            input_features, _ = prepare_features_for_model(features, batch_size)
            
            print(file_path, "\n", input_features)
            
            input_features = torch.tensor(input_features, dtype=torch.float32)
            input_features = input_features.view(1, -1)  # Reshape if necessary

            with torch.no_grad():
                logits = model(input_features)
                predictions = torch.argmax(logits, dim=1)  # Assuming classification task

            print(filename, "Predictions:", predictions)


            activations = input_features[0][1]
            parameters = input_features[0][3]
            batch_size = input_features[0][4]
            gradients = parameters

            horus_formula_estimation = (activations * batch_size + parameters) + (batch_size * gradients)
            horus_in_bytes = horus_formula_estimation * 4

            horus_estimations_MB = horus_in_bytes / (1024 ** 2)

            print("Horus Formual Estimation: ", horus_estimations_MB, activations, parameters, batch_size)


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

    # Packing the features into a list for transformer-based cnn estimator (modify as needed)
    feature_list_transformer = [
        depth,                                   # Feature 1
        total_activations,                       # Feature 2
        total_activations * batch_size,          # Feature 3
        total_params,                            # Feature 4
        batch_size,                              # Feature 5
        layer_counts['Linear'],                  # Feature 6
        layer_counts['LayerNorm'],               # Feature 7
        layer_counts['Dropout'],                 # Feature 8
    ]
    
    # Add other features as necessary for your model
    return feature_list_mlp, feature_list_transformer

# Usage example
if __name__ == "__main__":
    directory = 'Trans_models'  # Specify the directory containing .model files
    model_file = 'estimator/transformer_mlp_8gig.ckpt'  # Specify the path to the pickled model
    
    process_model_files(directory, model_file)