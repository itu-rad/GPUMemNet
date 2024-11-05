import os
import re
import csv
from collections import defaultdict

# Function to extract parameters from the filename based on known feature names
def extract_params_from_filename(filename):
    params = {}

    # Remove the '.out' suffix
    filename_base = filename.replace('.out', '')

    # Define regex patterns for each feature to match 'feature_name:value'
    patterns = {
        'num_classes': r'num_classes:(\d+)',
        'embedding_size': r'embedding_size:(\d+)',
        'num_layers': r'num_layers:(\d+)',
        'num_heads': r'num_heads:(\d+)',
        'ff_hidden_size': r'ff_hidden_size:(\d+)',
        'dropout_rate': r'dropout_rate:([\d\.]+)',  # Handling float numbers
        'seq_length': r'seq_length:(\d+)',
        'input_size': r'input_size:(\d+)',
        'num_samples': r'num_samples:(\d+)',
        'batch_size': r'batch_size:(\d+)',
    }

    # Iterate through patterns and apply them to extract values
    for key, pattern in patterns.items():
        match = re.search(pattern, filename_base)
        if match:
            # Convert matched values to appropriate type
            if key == 'dropout_rate':  # Handle float conversion for dropout_rate
                params[key] = float(match.group(1))
            else:
                params[key] = int(match.group(1))
        else:
            print(f"Warning: {key} not found in filename: {filename}")

    return params

# Function to extract model information from a transformer summary
def extract_transformer_model_info(summary, batch_size, sequence_length):
    if not summary.strip():  # Check if the summary is empty
        return None

    lines = summary.split('\n')

    activations_params = []
    total_params = 0
    total_activations = 0
    accumulated_params = 0  # Accumulated parameters over transformer layers
    depth = 0  # Count of layers
    gpu_memory_for_activations = 0  # Track memory for activations

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

    for line in lines:
        # print(line)

        total_params_match = re.search(r'Total params:\s*([\d,]+)', line)
        if total_params_match:
            total_params_str = total_params_match.group(1).replace(',', '')  # Remove commas
            total_params = int(total_params_str)

        # check for Embedding layer
        embedding_match = re.search(embedding_pattern, line)
        if embedding_match:
            vocab_size, embedding_dim, params = embedding_match.groups()
            params = int(params.replace(',', ''))  # Remove commas and convert to int
            accumulated_params += params  # Accumulate over the transformer layers

            # Calculate the number of activations for the embedding layer
            current_activations = sequence_length * int(embedding_dim)

            # Append the layer and its activations to the list
            activations_params.append(('Embedding', current_activations * batch_size , params))
            total_activations += current_activations

            # Increment layer count
            layer_counts['Embedding'] += 1
            depth += 1

            # print(current_activations)

            # Skip further checks for this line since it's already processed
            continue

        # Now check for NonDynamicallyQuantizableLinear
        non_dyn_linear_match = re.search(non_dynamically_linear_pattern, line)
        if non_dyn_linear_match:
            params = int(non_dyn_linear_match.group(1).replace(',', ''))
            accumulated_params += params  # Accumulate over the transformer layers

            # Extract in_features and out_features to update current_activations for NonDynamicallyQuantizableLinear layers
            activation_match = re.search(activations_pattern, line)
            if activation_match:
                in_features, out_features = map(int, activation_match.groups())
                current_activations = out_features   # Update the activations


            # Append the layer and its activations to the list
            # I figured it out that "NonDynamicallyQuantizableLinear" is same as "Linear"
            activations_params.append(('Linear', current_activations * batch_size, params))
            total_activations += current_activations

            # Increment layer count
            layer_counts['NonDynamicallyQuantizableLinear'] += 1
            layer_counts['Linear'] += 1
            depth += 1

            # print(current_activations)
            # Skip further checks for this line since it's already processed
            continue

        # Now check for Linear layer information
        linear_match = re.search(linear_pattern, line)
        if linear_match:
            params = int(linear_match.group(1).replace(',', ''))
            accumulated_params += params  # Accumulate over the transformer layers

            # Extract in_features and out_features to update current_activations for Linear layers
            activation_match = re.search(activations_pattern, line)
            if activation_match:
                in_features, out_features = map(int, activation_match.groups())
                current_activations = out_features   # Update the activations

                # print(current_activations)

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
                params = 0  # Default to zero if not found

            # print(current_activations)
            
            layer_counts['LayerNorm'] += 1
            activations_params.append(('LayerNorm', current_activations, params))  # No params for LayerNorm
            total_activations += current_activations
            depth += 1

        elif 'Dropout' in line:
            layer_counts['Dropout'] += 1
            activations_params.append(('Dropout', current_activations, 0))  # No params for Dropout
            total_activations += current_activations
            depth += 1

            # print(current_activations)

    return {
        "activations_params": activations_params,
        "total_params": total_params,
        "total_activations": total_activations,
        "accumulated_params": accumulated_params,
        "layer_counts": dict(layer_counts),
        "depth": depth
    }


# Function to check for Out-Of-Memory errors in the .err files
def check_for_oom_error(err_file):
    if os.path.exists(err_file):
        with open(err_file, 'r') as file:
            for line in file:
                if "torch.OutOfMemoryError: CUDA out of memory." in line:
                    return "OOM_CRASH"
    return "SUCCESSFUL"

# Extract GPU memory usage from nvsm.txt file
def extract_gpu_memory(nvsm_file):
    max_memory = 0
    target_gpu = "GPU-00f900e0-bb6f-792a-1b8a-597214c7e1a1"  # Example target GPU
    if os.path.exists(nvsm_file):
        with open(nvsm_file, 'r') as file:
            for line in file:
                if target_gpu in line:
                    match = re.findall(rf'{target_gpu},\s*(\d+)\s*MiB', line)
                    if match:
                        memory_used = int(match[0])
                        max_memory = max(max_memory, memory_used)
    return max_memory

# Process the entire dataset folder
def process_dataset(directories, output_csv):
    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['Filename', 'Batch Size', 'Seq Length', 'Embedding Size', 'Num Layers', 'Num Heads', 
                      'Depth', 'Accumulated Params', 'Activations-Params', 'Total Activations', 'Total Parameters', 
                      'Max GPU Memory (MiB)', 'NonDynamicallyQuantizableLinear Count', 'Linear Count', 'LayerNorm Count', 'Dropout Count', 'Status']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_dir in directories:
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    if file.endswith('.out'):
                        out_file_path = os.path.join(root, file)

                        # Extract parameters from filename
                        params = extract_params_from_filename(file)
                        batch_size = params.get('batch_size', 1)
                        sequence_length = params.get('seq_length', 1)
                        embedding_size = params.get('embedding_size', 0)
                        num_layers = params.get('num_layers', 0)
                        num_heads = params.get('num_heads', 0)

                        # Check for corresponding .err file
                        err_file_path = out_file_path.replace('.out', '.err')
                        status = check_for_oom_error(err_file_path)  # Check status based on the .err file

                        # Extract model info from .out file
                        with open(out_file_path, 'r') as out_file:
                            summary = out_file.read()

                        # Skip empty summaries
                        model_info = extract_transformer_model_info(summary, batch_size, sequence_length)
                        if model_info is None:  # If the file is empty, skip this iteration
                            print(f"Skipping empty .out file: {file}")
                            continue

                        # Extract GPU memory usage
                        nvsm_file_path = out_file_path.replace('.out', '_nvsm.txt')
                        max_gpu_memory = extract_gpu_memory(nvsm_file_path) if os.path.exists(nvsm_file_path) else None

                        # Write to CSV
                        writer.writerow({
                            'Filename': file,
                            'Batch Size': batch_size,
                            'Seq Length': sequence_length,
                            'Embedding Size': embedding_size,
                            'Num Layers': num_layers,
                            'Num Heads': num_heads,
                            'Depth': model_info['depth'],
                            'Accumulated Params': model_info['accumulated_params'],
                            'Activations-Params': model_info['activations_params'],
                            'Total Activations': model_info['total_activations'],
                            'Total Parameters': model_info['total_params'],
                            'Max GPU Memory (MiB)': max_gpu_memory,
                            'NonDynamicallyQuantizableLinear Count': model_info['layer_counts'].get('NonDynamicallyQuantizableLinear', 0),
                            'Linear Count': model_info['layer_counts'].get('Linear', 0),
                            'LayerNorm Count': model_info['layer_counts'].get('LayerNorm', 0),
                            'Dropout Count': model_info['layer_counts'].get('Dropout', 0),
                            'Status': status
                        })

                        break

if __name__ == "__main__":
    dataset_directories = ["../../transformer_dataset_step1", "../../transformer_dataset_step2"]  # Replace with actual dataset directories
    output_csv_path = "transformer_data_m2.csv"  # Output CSV file
    
    process_dataset(dataset_directories, output_csv_path)