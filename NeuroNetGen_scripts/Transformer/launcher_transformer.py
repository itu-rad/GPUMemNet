import subprocess
import os
import time
import numpy as np

# Directory to save the data logs and outputs
base_data_dir = "/home/ehyo/01-new_approach_dataset_fc/transformer_dataset_step1"
os.makedirs(base_data_dir, exist_ok=True)

# Number of random configurations to generate
num_random_configs = 3000  # You can change this to any number of configurations

# Seed for reproducibility
np.random.seed(99)

# Function to generate random Transformer parameters
def generate_random_transformer_config():
    # Choose a number of classes between 2 and 1000, which is a reasonable range for classification tasks.
    num_classes = int(np.random.uniform(2, 1001))  
    
    # Base embedding size should typically be in the range [128, 1024] for most transformer architectures.
    embedding_size = int(np.random.choice([128, 256, 512, 768, 1024]))  
    
    # Number of transformer layers: reasonable values range from 2 to 12 for small to medium models.
    num_layers = int(np.random.uniform(2, 13))  
    
    # Number of attention heads: usually a divisor of the embedding size, typically ranging from 2 to 16.
    num_heads = int(np.random.choice([2, 4, 8, 12, 16]))  
    
    # Feed-forward hidden size is often 2 to 4 times the embedding size; here we use a range based on the embedding size.
    ff_hidden_size = int(np.random.uniform(2 * embedding_size, 4 * embedding_size))  
    
    # Dropout rate between 0.1 and 0.5 is common in transformer architectures to prevent overfitting.
    dropout_rate = np.random.uniform(0.1, 0.5)  
    
    # Sequence length: typical values for transformers are 128, 256, or 512. We keep it within a sensible range.
    seq_length = int(np.random.choice([128, 256, 512]))  
    
    # Input size for vocabulary: using a vocabulary size between 50,000 and 1,000,000 is common for transformer models.
    input_size = int(np.random.uniform(50000, 1000000))  
    
    # Randomly generate number of samples (1000 to 5000), a range commonly used in transformer datasets.
    num_samples = int(np.random.uniform(1000, 5001))  
    
    # Randomly generate batch size (8 to 64) as a common practice to balance memory usage and training speed.
    batch_size = int(np.random.choice([8, 16, 32, 64]))  
    
    return num_classes, embedding_size, num_layers, num_heads, ff_hidden_size, dropout_rate, seq_length, input_size, num_samples, batch_size

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

# Function to run the transformer model training and monitoring
def run_transformer_experiment(config_name, num_classes, embedding_size, num_layers, num_heads, ff_hidden_size, dropout_rate, seq_length, input_size, num_samples, batch_size):
    print(f"Processing config: {config_name}, num classes: {num_classes}, embedding size: {embedding_size}, num layers: {num_layers}, num heads: {num_heads}, ff hidden size: {ff_hidden_size}, dropout rate: {dropout_rate}, seq length: {seq_length}, input size: {input_size}, num samples: {num_samples}, batch size: {batch_size}")

    # Create a subdirectory for each configuration inside the base directory
    config_dir = os.path.join(base_data_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)

    # Define the file suffix using the transformer parameters concatenated with underscores
    file_suffix = f"num_classes:{num_classes}_embedding_size:{embedding_size}_num_layers:{num_layers}_num_heads:{num_heads}_ff_hidden_size:{ff_hidden_size}_dropout_rate:{dropout_rate}_seq_length:{seq_length}_input_size:{input_size}_num_samples:{num_samples}_batch_size:{batch_size}"

    # Define the output and error file paths inside the configuration's directory
    out_file = os.path.join(config_dir, f"{file_suffix}.out")
    err_file = os.path.join(config_dir, f"{file_suffix}.err")

    # Run monitoring tools
    nvidia_smi_proc, dcgmi_proc, top_proc = run_monitoring_tools(config_dir, file_suffix)

    # Command to run the transformer training
    train_cmd = f"python transformer.py --input_size {input_size} --embedding_size {embedding_size} --num_layers {num_layers} --num_heads {num_heads} --ff_hidden_size {ff_hidden_size} --dropout_rate {dropout_rate} --seq_length {seq_length} --output_size {num_classes} --num_samples {num_samples} --batch_size {batch_size}"

    # Execute the transformer training and redirect output and error
    with open(out_file, "w") as out_f, open(err_file, "w") as err_f:
        train_process = subprocess.Popen(train_cmd, shell=True, stdout=out_f, stderr=err_f)
        train_process.wait()  # Wait for the training process to complete

    # Kill monitoring tools
    kill_monitoring_tools(nvidia_smi_proc, dcgmi_proc, top_proc)

# Main function to run experiments with random transformer configurations
def main():
    for i in range(1, num_random_configs + 1):
        # Generate a random transformer configuration
        num_classes, embedding_size, num_layers, num_heads, ff_hidden_size, dropout_rate, seq_length, input_size, num_samples, batch_size = generate_random_transformer_config()

        # Create a folder with a zero-padded index like "01-transformer_config"
        config_name = f"{i:02d}-transformer_config"

        # Run the experiment for each configuration
        run_transformer_experiment(config_name, num_classes, embedding_size, num_layers, num_heads, ff_hidden_size, dropout_rate, seq_length, input_size, num_samples, batch_size)

if __name__ == "__main__":
    main()