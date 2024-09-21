import os
import re
import csv
from collections import Counter

def most_frequent_activation_function(activation_functions):
    # Count the occurrences of each activation function
    activation_counter = Counter(activation_functions)
    
    # Find the activation function with the highest count
    # most_common_activation, count = activation_counter.most_common(1)[0]
    most_common_activation, _ = activation_counter.most_common(1)[0]
    
    return most_common_activation

# Function to check for OOM errors in the corresponding .err file
def check_for_oom_error(err_file):
    try:
        if os.path.exists(err_file):
            with open(err_file, 'r') as file:
                for line in file:
                    if "torch.OutOfMemoryError: CUDA out of memory." in line:
                        return "OOM_CRASH"
        return "SUCCESSFUL"
    except Exception as e:
        print(f"Error checking {err_file}: {e}")
        return "UNKNOWN_ERROR"


# Extract model info from the .out file (Torch summary)
# Extract model info from the .out file (Torch summary)
def extract_model_info(out_file):
    try:
        with open(out_file, 'r') as file:
            lines = file.readlines()

        if not lines:  # If the file is empty
            return [], 'None', 0, 0, 0, {
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

        # Dictionary to keep track of the count of each layer type
        layer_counts = {
            'conv2d': 0,
            'batchnorm2d': 0,
            'dropout': 0,
            'adaptive_avg_pool2d': 0,
            'linear': 0,
            'softmax': 0
        }

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
                    activations_params.append(('conv2d', activations, params))
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
                    activations_params.append(('batchnorm2d', activations, params))
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
                    activations_params.append(('dropout', activations, 0))
                    total_activations += activations
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
                    activations_params.append(('adaptive_avg_pool2d', activations, 0))  # No parameters for pooling
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
                    activations_params.append(('linear', activations, params))
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
                    activations_params.append(('softmax', activations, 0))  # Softmax has no trainable parameters
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
                    activations_params.append((activation_func, activations, 0))  # No parameters for activation functions
                    total_activations += activations
                    activation_functions_list.append(activation_func)
                else:
                    print(f"Skipping layer: {line.strip()} (no activation data)")

        activation_function = most_frequent_activation_function(activation_functions_list)

        return activations_params, activation_function, depth, total_params, total_activations, layer_counts

    except Exception as e:
        print(f"Error processing {out_file}: {e}")
        return [], 'None', 0, 0, 0, {
            'conv2d': 0,
            'batchnorm2d': 0,
            'dropout': 0,
            'adaptive_avg_pool2d': 0,
            'linear': 0,
            'softmax': 0
        }

# Extract batch size from the filename
def extract_batch_size(filename):
    match = re.search(r'batch:(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

# Extract the maximum GPU memory used for the specific GPU from the nvsm.txt file
def extract_gpu_memory(nvsm_file):
    max_memory = 0
    target_gpu = "GPU-00f900e0-bb6f-792a-1b8a-597214c7e1a1"
    
    with open(nvsm_file, 'r') as file:
        for line in file:
            if target_gpu in line:
                # Extract the memory used value for the target GPU
                match = re.findall(rf'{target_gpu},\s*(\d+)\s*MiB', line)
                if match:
                    memory_used = int(match[0])
                    max_memory = max(max_memory, memory_used)
    
    return max_memory

# Extract DCGMI data for GPU 0 from the dcgm.txt file
def extract_and_average_dcgmi(file_path):
    # Initialize lists for storing values of GPUTL, GRACT, SMACT, SMOCC, FP32A
    gputl_values = []
    gract_values = []
    smact_values = []
    smocc_values = []
    fp32a_values = []

    try:
        # Check if the file is empty
        if os.path.getsize(file_path) == 0:
            return -1, -1, -1, -1, -1  # Return None if the file is empty
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            # Look for GPU 0 rows
            if line.startswith("GPU 0"):
                values = re.split(r'\s+', line.strip())

                # Extract values for GPUTL (index 2), GRACT (index 3), SMACT (index 4), SMOCC (index 5), FP32A (index 7)
                gputl = float(values[2])
                gract = float(values[3])
                smact = float(values[4])
                smocc = float(values[5])
                fp32a = float(values[7])

                # Add non-zero values to corresponding lists
                if gputl != 0:
                    gputl_values.append(gputl)
                if gract != 0:
                    gract_values.append(gract)
                if smact != 0:
                    smact_values.append(smact)
                if smocc != 0:
                    smocc_values.append(smocc)
                if fp32a != 0:
                    fp32a_values.append(fp32a)

        # Calculate the averages for each metric, handle division by zero
        avg_gputl = sum(gputl_values) / len(gputl_values) if gputl_values else 0
        avg_gract = sum(gract_values) / len(gract_values) if gract_values else 0
        avg_smact = sum(smact_values) / len(smact_values) if smact_values else 0
        avg_smocc = sum(smocc_values) / len(smocc_values) if smocc_values else 0
        avg_fp32a = sum(fp32a_values) / len(fp32a_values) if fp32a_values else 0

        # Return the averages
        return avg_gputl, avg_gract, avg_smact, avg_smocc, avg_fp32a
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    

# Process the entire dataset folder
def process_dataset(dataset_dir, output_csv):
    with open(output_csv, mode='w', newline='') as csv_file:
        # Add new fields for layer counts in the fieldnames list
        fieldnames = ['Filename', 'Depth', 'Activations-Params', 'Activation Function', 'Total Activations', 'Total Parameters', 'Batch Size', 
                      'Max GPU Memory (MiB)', 'Avg GPUTL', 'Avg GRACT', 'Avg SMACT', 'Avg SMOCC', 'Avg FP32A', 
                      'Conv2d Count', 'BatchNorm2d Count', 'Dropout Count', 'AdaptiveAvgPool2d Count', 'Linear Count', 'Status']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('.out'):
                    out_file_path = os.path.join(root, file)

                    # Check for corresponding .err file
                    err_file_path = out_file_path.replace('.out', '.err')
                    status = check_for_oom_error(err_file_path)  # Check status based on the .err file

                    # Extract model info
                    activations_params, activation_function, depth, total_params, total_activations, layer_counts = extract_model_info(out_file_path)
                    batch_size = extract_batch_size(file)

                    
                    # Get the corresponding nvsm file
                    nvsm_file_pattern = file.replace('.out', '_nvsm.txt')
                    nvsm_file_path = os.path.join(root, nvsm_file_pattern)
                    
                    if os.path.exists(nvsm_file_path):
                        max_gpu_memory = extract_gpu_memory(nvsm_file_path)
                    else:
                        max_gpu_memory = None

                    # Get the corresponding dcgm file
                    dcgm_file_pattern = file.replace('.out', '_dcgm.txt')
                    dcgm_file_path = os.path.join(root, dcgm_file_pattern)

                    if os.path.exists(dcgm_file_path):
                        avg_gputl, avg_gract, avg_smact, avg_smocc, avg_fp32a = extract_and_average_dcgmi(dcgm_file_path)
                    else:
                        avg_gputl, avg_gract, avg_smact, avg_smocc, avg_fp32a = {col: None for col in ["GPUTL", "GRACT", "SMACT", "SMOCC", "FP32A"]}
                        
                    writer.writerow({
                        'Filename': file,
                        'Depth': depth,
                        'Activations-Params': activations_params,
                        'Activation Function': activation_function,
                        'Total Activations': total_activations,
                        'Total Parameters': total_params,
                        'Batch Size': batch_size,
                        'Max GPU Memory (MiB)': max_gpu_memory,
                        'Avg GPUTL': avg_gputl,
                        'Avg GRACT': avg_gract,
                        'Avg SMACT': avg_smact,
                        'Avg SMOCC': avg_smocc,
                        'Avg FP32A': avg_fp32a,
                        'Conv2d Count': layer_counts['conv2d'],
                        'BatchNorm2d Count': layer_counts['batchnorm2d'],
                        'Dropout Count': layer_counts['dropout'],
                        'AdaptiveAvgPool2d Count': layer_counts['adaptive_avg_pool2d'],
                        'Linear Count': layer_counts['linear'],
                        'Status': status
                    })

if __name__ == "__main__":
    dataset_directory = "cnn_dataset_step1"  # Replace with the actual dataset directory path
    output_csv_path = "cnn_data_step1.csv"  # Specify the desired output CSV file
    process_dataset(dataset_directory, output_csv_path)