import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from tqdm import tqdm
from torchsummary import summary
import numpy as np
from multiprocessing import Pool, cpu_count

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, architecture='pyramid'):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size
        # Pyramid or tapered structure
        if architecture == 'pyramid':
            for i in range(hidden_layers):
                out_features = max(int(in_features * 0.5), output_size)
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.ReLU())
                in_features = out_features
        # Uniform layer structure
        elif architecture == 'uniform':
            for i in range(hidden_layers):
                layers.append(nn.Linear(in_features, in_features))
                layers.append(nn.ReLU())
        # Bottleneck: reducing neurons
        elif architecture == 'bottleneck':
            for i in range(hidden_layers):
                out_features = max(int(in_features * 0.5), output_size)
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.ReLU())
                in_features = out_features
            layers.append(nn.Linear(in_features, 64))
            layers.append(nn.ReLU())
            in_features = 64
        # Gradual reduction
        elif architecture == 'gradual':
            decrement = (input_size - output_size) // hidden_layers
            for i in range(hidden_layers):
                out_features = max(in_features - decrement, output_size)
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.ReLU())
                in_features = out_features
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def generate_labels(output_size, num_samples):
    if output_size == 1:
        return torch.randint(0, 2, (num_samples,))
    else:
        return torch.randint(0, output_size, (num_samples,))

def generate_dummy_dataset(input_size, output_size, num_samples, batch_size):
    inputs = torch.randn(num_samples, input_size)
    # Use multiprocessing to speed up label generation
    num_cores = min(cpu_count(), num_samples)  # Use as many cores as there are samples, or max available
    chunk_size = num_samples // num_cores
    
    pool = Pool(num_cores)
    # Generate labels in parallel and ensure they match the total number of samples
    labels_chunks = pool.starmap(generate_labels, [(output_size, chunk_size) for _ in range(num_cores)])
    
    # Handle any remaining samples (remainder if num_samples is not divisible by num_cores)
    remainder = num_samples % num_cores
    if remainder:
        labels_chunks.append(generate_labels(output_size, remainder))
    
    labels = torch.cat(labels_chunks)
    pool.close()
    pool.join()
    
    # Ensure the inputs and labels have matching sizes
    assert len(inputs) == len(labels), f"Size mismatch: inputs ({len(inputs)}) and labels ({len(labels)})"
    
    dataset = TensorDataset(inputs, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def train(model, train_loader, device, time_limit=180):
    criterion = nn.CrossEntropyLoss() if model.model[-1].out_features > 1 else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    model.to(device)
    start_time = time.time()
    epoch = 0
    progress_bar = tqdm(total=100, desc="Training Progress", position=0, leave=True)

    while True:
        epoch_loss = 0.0
        total_batches = len(train_loader)
        with tqdm(total=total_batches, desc=f'Epoch {epoch + 1}', unit='batch', leave=False) as pbar:
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if time.time() - start_time > time_limit:
                    print(f"Training stopped due to time limit ({time_limit} seconds)")
                    progress_bar.close()  # Close the progress bar when stopping
                    return
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                
                # Update the main progress bar based on time elapsed
                time_progress = (time.time() - start_time) / time_limit
                progress_bar.n = min(int(time_progress * 100), 100)
                progress_bar.refresh()

        print(f"Epoch {epoch + 1} complete. Average Loss: {epoch_loss / total_batches:.4f}")
        epoch += 1
        progress_bar.update(1)

    progress_bar.close()  # Ensure progress bar is closed when training ends

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a configurable MLP with different architectures on a dummy dataset.')
    parser.add_argument('--depth', type=int, default=3, help='Number of hidden layers (depth of the MLP)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--input_size', type=int, required=True, help='Input size (number of features)')
    parser.add_argument('--output_size', type=int, required=True, help='Output size (number of classes or targets)')
    parser.add_argument('--num_samples', type=int, default=4000, help='Number of samples in the dummy dataset')
    parser.add_argument('--architecture', type=str, choices=['pyramid', 'uniform', 'bottleneck', 'gradual'], default='pyramid', help='Architecture type for the MLP')
    args = parser.parse_args()
    train_loader = generate_dummy_dataset(args.input_size, args.output_size, args.num_samples, args.batch_size)
    model = MLP(args.input_size, args.output_size, args.depth, args.architecture).to(device)
    summary(model, input_size=(1, args.input_size), device=device)
    train(model, train_loader, device)