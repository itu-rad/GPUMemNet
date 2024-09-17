import subprocess
import os
import time
import numpy as np
import multiprocessing

# Directory to save the data logs and outputs
base_data_dir = "cnn_dataset_step1"
os.makedirs(base_data_dir, exist_ok=True)

# Number of random configurations to generate
num_random_configs = 3000  # You can change this to any number of configurations

# Seed for reproducibility
np.random.seed(99)

# Function to generate random CNN parameters using NumPy's random.uniform
def generate_random_cnn_config():
    input_channels = int(np.random.choice([1, 3]))  # Randomly select between grayscale (1) and RGB (3) channels
    num_classes = int(np.random.uniform(2, 22000))  # Random number of output classes (2-100 for classification)
    base_num_filters = int(np.random.uniform(16, 512))  # Base number of filters between 16 and 512
    filter_size = int(np.random.choice([3, 5, 7]))  # Random filter size from common values (3x3, 5x5, 7x7)
    depth = int(np.random.uniform(1, 200))  # Random depth (number of convolutional layers)
    architecture = np.random.choice(['pyramid', 'uniform', 'bottleneck', 'gradual', 'hourglass', 'residual', 'dense'])  # Random architecture type
    
    # Randomly choose activation function
    activation = np.random.choice(['relu', 'leaky_relu', 'prelu', 'elu', 'selu', 'tanh', 'softplus', 'swish', 'mish', 'gelu'])
    
    # Random dropout rate
    dropout_rate = np.random.uniform(0.1, 0.5)  # Random dropout rate between 0.1 and 0.5

    # Randomly decide whether to use dropout, batch normalization, skip connections, and dilated convolutions
    use_dropout = np.random.choice([True, False])
    use_batch_norm = np.random.choice([True, False])
    use_skip = np.random.choice([True, False])
    use_dilated = np.random.choice([True, False])
    use_depthwise_separable = np.random.choice([True, False])
    
    # Input size: common image sizes (e.g., 32x32, 128x128, 224x224, etc.)
    input_size = int(np.random.choice([28, 32, 64, 128, 224, 299, 334, 512, 1024]))

    batch_size = int(np.random.uniform(1, 256) * 4)   # Batch size (multiple of 4)

    return input_channels, num_classes, depth, architecture, base_num_filters, filter_size, batch_size, input_size, activation, dropout_rate, use_dropout, use_batch_norm, use_skip, use_dilated, use_depthwise_separable

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

# Function to run the CNN model training and monitoring
def run_cnn_experiment(config_name, input_channels, num_classes, depth, architecture, base_num_filters, filter_size, batch_size, input_size, activation, dropout_rate, use_dropout, use_batch_norm, use_skip, use_dilated, use_depthwise_separable):
    print(f"Processing config: {config_name}, input channels: {input_channels}, num classes: {num_classes}, depth: {depth}, architecture: {architecture}, batch size: {batch_size}")

    # Create a subdirectory for each configuration inside the base directory
    config_dir = os.path.join(base_data_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)

    # Define the file suffix using the CNN parameters concatenated with underscores
    file_suffix = f"input_channels:{input_channels}_num_classes:{num_classes}_depth:{depth}_arch:{architecture}_base_filters:{base_num_filters}_filter_size:{filter_size}_batch:{batch_size}_input_size:{input_size}_act:{activation}_dropout:{dropout_rate}_dropout:{use_dropout}_batchnorm:{use_batch_norm}_skip:{use_skip}_dilated:{use_dilated}_depthwise:{use_depthwise_separable}"

    # Define the output and error file paths inside the configuration's directory
    out_file = os.path.join(config_dir, f"{file_suffix}.out")
    err_file = os.path.join(config_dir, f"{file_suffix}.err")

    # Run monitoring tools
    nvidia_smi_proc, dcgmi_proc, top_proc = run_monitoring_tools(config_dir, file_suffix)

    # Command to run the CNN training
    train_cmd = f"CUDA_VISIBLE_DEVICES=0 python cnn_next.py --input_channels {input_channels} --num_classes {num_classes} --depth {depth} --architecture {architecture} --base_num_filters {base_num_filters} --filter_size {filter_size} --batch_size {batch_size} --input_size {input_size} --activation {activation} --dropout_rate {dropout_rate} {'--use_dropout' if use_dropout else ''} {'--use_batch_norm' if use_batch_norm else ''} {'--use_skip' if use_skip else ''} {'--use_dilated' if use_dilated else ''} {'--use_depthwise_separable' if use_depthwise_separable else ''}"

    # Execute the CNN training and redirect output and error
    with open(out_file, "w") as out_f, open(err_file, "w") as err_f:
        train_process = subprocess.Popen(train_cmd, shell=True, stdout=out_f, stderr=err_f)
        train_process.wait()  # Wait for the training process to complete

    # Kill monitoring tools
    kill_monitoring_tools(nvidia_smi_proc, dcgmi_proc, top_proc)

# Main function to run experiments with random CNN configurations
def main():
    for i in range(1, num_random_configs + 1):
        # Generate a random CNN configuration using NumPy's random.uniform
        input_channels, num_classes, depth, architecture, base_num_filters, filter_size, batch_size, input_size, activation, dropout_rate, use_dropout, use_batch_norm, use_skip, use_dilated, use_depthwise_separable = generate_random_cnn_config()

        # Create a folder with a zero-padded index like "01-cnn_config"
        config_name = f"{i:02d}-cnn_config"

        # Run the experiment for each configuration
        run_cnn_experiment(config_name, input_channels, num_classes, depth, architecture, base_num_filters, filter_size, batch_size, input_size, activation, dropout_rate, use_dropout, use_batch_norm, use_skip, use_dilated, use_depthwise_separable)

if __name__ == "__main__":
    main()
