import os
import pickle
import re

from collections import Counter


def most_frequent_activation_function(activation_functions):
    # Count the occurrences of each activation function
    activation_counter = Counter(activation_functions)
    
    # Find the activation function with the highest count
    # most_common_activation, count = activation_counter.most_common(1)[0]
    most_common_activation, _ = activation_counter.most_common(1)[0]
    
    return most_common_activation

# Extract model info from the .out file (Torch summary)
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
            if "Block" in line:
                continue

            # Handle Conv2d layers
            if "Conv2d-" in line:
                depth += 1
                layer_counts['conv2d'] += 1
                output_shape = re.findall(r'\[\-1, (\d+), (\d+), (\d+)\]', line)
                param_count = re.findall(r'(\d{1,3}(?:,\d{3})*)$', line)
                if output_shape and param_count:
                    channels = int(output_shape[0][0])
                    height = int(output_shape[0][1])
                    width = int(output_shape[0][2])
                    activations = channels * height * width
                    params = int(param_count[0].replace(',', ''))

                    # print(line, activations, params, f"{height}, {width}, {channels}")

                    activations_params.append(('conv2d', activations * batch_size, params))
                    total_params += params
                    total_activations += activations
                else:
                    print(f"Warning: Missing Conv2d data in {out_file}, line: {line.strip()}")

            # Handle BatchNorm2d layers
            elif "BatchNorm2d-" in line:
                layer_counts['batchnorm2d'] += 1
                output_shape = re.findall(r'\[\-1, (\d+), (\d+), (\d+)\]', line)
                param_count = re.findall(r'(\d{1,3}(?:,\d{3})*)$', line)

                if output_shape and param_count:
                    channels = int(output_shape[0][0])
                    height = int(output_shape[0][1])
                    width = int(output_shape[0][2])
                    activations = channels * height * width
                    params = int(param_count[0].replace(',', ''))

                    # print(line, activations, params, f"{height}, {width}, {channels}")

                    activations_params.append(('batchnorm2d', activations * batch_size, params))
                    total_params += params
                    total_activations += activations
                else:
                    print(f"Warning: Missing BatchNorm2d data in {out_file}, line: {line.strip()}")

            # Handle Dropout layers
            elif "Dropout-" in line:
                layer_counts['dropout'] += 1
                output_shape = re.findall(r'\[\-1, (\d+), (\d+), (\d+)\]', line)
                if output_shape:
                    channels = int(output_shape[0][0])
                    height = int(output_shape[0][1])
                    width = int(output_shape[0][2])
                    activations = channels * height * width
                    activations_params.append(('dropout', activations * batch_size, 0))
                    total_activations += activations

                    # print(line, activations, params, f"{height}, {width}, {channels}")
                else:
                    print(f"Warning: Missing Dropout data in {out_file}, line: {line.strip()}")

            # Handle AdaptiveAvgPool2d layers, including small output shapes like [1,1]
            elif "AdaptiveAvgPool2d-" in line:
                layer_counts['adaptive_avg_pool2d'] += 1
                output_shape = re.findall(r'\[\-1, (\d+), (\d+), (\d+)\]', line)  # Typical shape
                if not output_shape:
                    output_shape = re.findall(r'\[\-1, (\d+), (\d+)\]', line)  # Fallback for shapes like [C, 1, 1]
                if not output_shape:
                    output_shape = re.findall(r'\[\-1, (\d+)\]', line)  # Fallback for shapes like [C]
                if output_shape:
                    channels = int(output_shape[0][0])
                    activations = channels  # AdaptiveAvgPool2d typically reduces spatial dimensions to 1x1
                    activations_params.append(('adaptive_avg_pool2d', activations * batch_size, 0))  # No parameters for pooling
                    
                    # print(line, activations, 0, f"{height}, {width}, {channels}")

                    total_activations += activations
                else:
                    print(f"Warning: Missing AdaptiveAvgPool2d data in {out_file}, line: {line.strip()}")

            # Handle Linear layers
            elif "Linear-" in line:
                layer_counts['linear'] += 1
                output_shape = re.findall(r'\[\-1, (\d+)\]', line)
                param_count = re.findall(r'(\d{1,3}(?:,\d{3})*)$', line)
                if output_shape and param_count:
                    activations = int(output_shape[0])
                    params = int(param_count[0].replace(',', ''))
                    activations_params.append(('linear', activations * batch_size, params))

                    # print(line, activations, params, f"{height}, {width}, {channels}")

                    total_params += params
                    total_activations += activations
                else:
                    print(f"Warning: Missing Linear data in {out_file}, line: {line.strip()}")

            # Handle Softmax layers
            elif "Softmax-" in line:
                layer_counts['softmax'] += 1
                output_shape = re.findall(r'\[\-1, (\d+)\]', line)  # Softmax typically has a 1D output
                if output_shape:
                    activations = int(output_shape[0])
                    activations_params.append(('softmax', activations * batch_size, 0))  # Softmax has no trainable parameters
                    total_activations += activations
                else:
                    print(f"Warning: Missing Softmax data in {out_file}, line: {line.strip()}")

            # Dynamically handle activation functions by identifying any name with common patterns
            elif re.search(r'(ReLU|LeakyReLU|PReLU|ELU|SELU|GELU|Tanh|SiLU|Softplus|Mish)-\d+', line):
                activation_func = re.findall(r'(ReLU|LeakyReLU|PReLU|ELU|SELU|GELU|Tanh|SiLU|Softplus|Mish)-\d+', line)[0]
                output_shape = re.findall(r'\[\-1, (\d+), (\d+), (\d+)\]', line)
                if output_shape:
                    channels = int(output_shape[0][0])
                    height = int(output_shape[0][1])
                    width = int(output_shape[0][2])
                    activations = channels * height * width
                    activations_params.append((activation_func, activations * batch_size, 0))  # No parameters for activation functions
                    total_activations += activations
                    activation_functions_list.append(activation_func)
                else:
                    print(f"Skipping layer: {line.strip()} (no activation data)")

            if "Total params:" in line:
                temp = re.findall(r'Total params:\s*([\d,]+)', line)

                # print("parameters by torch: ", temp)

                if temp:
                    temp = int(temp[0].replace(',', ''))
                    # print("temp again: ", temp)
            
            if "Input size (MB):" in line:
                input_size_mb = float(re.findall(r'Input size \(MB\):\s*([\d.]+)', line)[0])
                # print("input size detected: ", input_size_mb)
            elif "Forward/backward pass size (MB):" in line:
                forward_backward_size_mb = float(re.findall(r'Forward/backward pass size \(MB\):\s*([\d.]+)', line)[0])
            elif "Params size (MB):" in line:
                params_size_mb = float(re.findall(r'Params size \(MB\):\s*([\d.]+)', line)[0])
            elif "Estimated Total Size (MB):" in line:
                estimated_total_size_mb = float(re.findall(r'Estimated Total Size \(MB\):\s*([\d.]+)', line)[0])
                # break  # No need to continue parsing further since total params are found

        if temp != total_params:
            assert("PROBLEMMMMMMM.....")

        activation_function = most_frequent_activation_function(activation_functions_list)

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

def process_model_files(directory, model_file):
    # Load the pre-trained model from the pickled file
    with open(model_file, 'rb') as file:
        loaded_model = pickle.load(file)
    
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
            
            # Prepare the features to pass to the loaded model
            # (Adapt this part depending on how the model expects its input)
            input_features = prepare_features_for_model(features)
            
            # Make predictions using the loaded model
            predictions = loaded_model.predict([input_features])
            
            # Print the filename and predicted values
            print(f"File: {filename}, Batch Size: {batch_size}, Predictions: {predictions}")

def prepare_features_for_model(features):
    """
    Convert the features extracted by extract_model_info into the format
    required by the loaded model.
    
    Modify this function to extract relevant features.
    """
    # Unpack the features (adapt depending on what features your model needs)
    activations_params, activation_function, depth, total_params, total_activations, input_size_mb, forward_backward_size_mb, params_size_mb, estimated_total_size_mb, layer_counts = features
    
    # Example of packing the features into a list (modify as needed)
    feature_list = [
        total_params,
        total_activations,
        depth,
        input_size_mb,
        forward_backward_size_mb,
        params_size_mb,
        estimated_total_size_mb,
        layer_counts['conv2d'],
        layer_counts['batchnorm2d'],
        layer_counts['dropout'],
        layer_counts['adaptive_avg_pool2d'],
        layer_counts['linear'],
        layer_counts['softmax'],
    ]
    
    # Add other features as necessary for your model
    return feature_list

# Usage example
if __name__ == "__main__":
    directory = 'models'  # Specify the directory containing .model files
    model_file = 'estimator/model.pkl'  # Specify the path to the pickled model
    
    process_model_files(directory, model_file)
