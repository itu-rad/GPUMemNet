import subprocess
import os
import time
import numpy as np
import multiprocessing
import yaml


# Load config from YAML file
with open("mlp_config.yaml", "r") as file:
    config = yaml.safe_load(file)


# Directory to save the data logs and outputs
base_data_dir = config["base_data_dir"]
os.makedirs(base_data_dir, exist_ok=True)

# Number of random configurations to generate (instead of models_list.txt)
num_random_configs = config["num_random_configs"]  # You can change this to any number of configurations

# Seed for reproducibility
np.random.seed(99)

# Function to generate random MLP parameters using NumPy's random.uniform
def generate_random_mlp_config():
    input_size = int(np.random.uniform(config["input_size"]["min"], config["input_size"]["max"]))  # Random input size between 4 and 4096
    output_size = int(np.random.uniform(1, max(2, input_size * config["output_size"]["min_ratio"])))  # Random output size, smaller than input size
    batch_size = int(np.random.uniform(config["batch_size"]["min"], config["batch_size"]["max"]) * config["batch_size"]["multiplier"])  # batch_size
    depth = int(np.random.uniform(config["depth"]["min"], config["depth"]["max"]))  # Random depth (number of hidden layers)
    architecture = np.random.choice(config["architectures"])  # Random architecture type
    return input_size, output_size, depth, architecture, batch_size

# Function to run system monitoring tools (nvidia-smi, dcgmi, top)
def run_monitoring_tools(config_dir, file_suffix):
    file_name1 = os.path.join(config_dir, f"{file_suffix}_nvsm.txt")
    file_name2 = os.path.join(config_dir, f"{file_suffix}_dcgm.txt")
    file_name3 = os.path.join(config_dir, f"{file_suffix}_top.txt")

    # Run nvidia-smi
    nvidia_smi_cmd = f"nvidia-smi --query-gpu=uuid,memory.used,memory.total --format=csv -l 1 > {file_name1}"
    nvidia_smi_proc = subprocess.Popen(nvidia_smi_cmd, shell=True, preexec_fn=os.setsid)

    # Run dcgmi
    dcgmi_cmd = f"dcgmi dmon -e 203,1001,1002,1003,1006,1007,1008,1004,204,1005,1009,1010,1011,1012,155,156 > {file_name2}"
    dcgmi_proc = subprocess.Popen(dcgmi_cmd, shell=True, preexec_fn=os.setsid)

    # Run top
    top_cmd = f"top -i -b -n 999999999 > {file_name3}"
    top_proc = subprocess.Popen(top_cmd, shell=True, preexec_fn=os.setsid)

    return nvidia_smi_proc, dcgmi_proc, top_proc

# Function to kill monitoring processes
def kill_monitoring_tools(nvidia_smi_proc, dcgmi_proc, top_proc):
    os.killpg(os.getpgid(nvidia_smi_proc.pid), 9)
    os.killpg(os.getpgid(dcgmi_proc.pid), 9)
    os.killpg(os.getpgid(top_proc.pid), 9)

    nvidia_smi_proc.wait()
    dcgmi_proc.wait()
    top_proc.wait()

# Function to run the MLP model training and monitoring
def run_mlp_experiment(config_name, input_size, output_size, depth, architecture, batch_size):
    print(f"Processing config: {config_name}, input size: {input_size}, output size: {output_size}, depth: {depth}, architecture: {architecture}, Batch size: {batch_size}")

    # Create a subdirectory for each configuration inside the base directory
    config_dir = os.path.join(base_data_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)

    # Define the file suffix using the MLP parameters concatenated with underscores
    file_suffix = f"input:{input_size}_output:{output_size}_depth:{depth}_arch:{architecture}_batch:{batch_size}"

    # Define the output and error file paths inside the configuration's directory
    out_file = os.path.join(config_dir, f"{file_suffix}.out")
    err_file = os.path.join(config_dir, f"{file_suffix}.err")

    # Run monitoring tools
    nvidia_smi_proc, dcgmi_proc, top_proc = run_monitoring_tools(config_dir, file_suffix)

    # Command to run the MLP training
    train_cmd = f"CUDA_VISIBLE_DEVICES=0 python mlp.py --input_size {input_size} --output_size {output_size} --depth {depth} --architecture {architecture} --batch_size {batch_size}"

    # Execute the MLP training and redirect output and error
    with open(out_file, "w") as out_f, open(err_file, "w") as err_f:
        train_process = subprocess.Popen(train_cmd, shell=True, stdout=out_f, stderr=err_f)
        train_process.wait()  # Wait for the training process to complete

    # Kill monitoring tools
    kill_monitoring_tools(nvidia_smi_proc, dcgmi_proc, top_proc)

# Main function to run experiments with random MLP configurations
def main():
    for i in range(1, num_random_configs + 1):
        # Generate a random MLP configuration using NumPy's random.uniform
        input_size, output_size, depth, architecture, batch_size = generate_random_mlp_config()
        
        # Create a folder with a zero-padded index like "01-mlp_config"
        config_name = f"{i:02d}-mlp_config"

        # Run the experiment for each configuration
        run_mlp_experiment(config_name, input_size, output_size, depth, architecture, batch_size)

if __name__ == "__main__":
    main()