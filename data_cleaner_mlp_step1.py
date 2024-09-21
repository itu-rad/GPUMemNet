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

# Extract model info from the .out file (Torch summary)
def extract_model_info(out_file):
    with open(out_file, 'r') as file:
        lines = file.readlines()

    activations_params = []
    activation_functions = []
    total_params = 0
    total_activations = 0
    depth = 0

    for line in lines:
        if "Linear-" in line:
            depth += 1
            output_shape = re.findall(r'\[\-1, 1, (\d+)\]', line)
            param_count = re.findall(r'(\d{1,3}(?:,\d{3})*)$', line)
            if output_shape and param_count:
                activations = int(output_shape[0])
                params = int(param_count[0].replace(',', ''))
                activations_params.append(('linear', activations, params))
                total_params += params
                total_activations += activations

        elif "ReLU-" in line:
            output_shape = re.findall(r'\[\-1, 1, (\d+)\]', line)
            activation_func = re.findall(r'([A-Za-z]+)-\d+', line)
            if output_shape and activation_func:
                activations = int(output_shape[0])
                activations_params.append((activation_func[0], activations, 0))  # No parameters for activation layers
                total_activations += activations
                activation_functions.append(activation_func[0])  # Add the activation function used

    activation_function = most_frequent_activation_function(activation_functions)

    # print(activation_function)
    
    return activations_params, activation_function, depth, total_params, total_activations

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

# Process the entire dataset folder
def process_dataset(dataset_dir, output_csv):
    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['Filename', 'Depth', 'Activations-Params', 'Activation Function', 'Total Activations', 'Total Parameters', 'Batch Size', 
                      'Max GPU Memory (MiB)', 'Avg GPUTL', 'Avg GRACT', 'Avg SMACT', 'Avg SMOCC', 'Avg FP32A']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('.out'):
                    out_file_path = os.path.join(root, file)
                    activations_params, activation_function, depth, total_params, total_activations = extract_model_info(out_file_path)
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
                        'Avg FP32A': avg_fp32a
                    })

if __name__ == "__main__":
    dataset_directory = "mlp_dataset_step1"  # Replace with the actual dataset directory path
    output_csv_path = "mlp_data_step1.csv"  # Specify the desired output CSV file
    process_dataset(dataset_directory, output_csv_path)