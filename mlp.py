import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
from torchsummary import summary
from tqdm import tqdm  # Import tqdm for progress bars

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, width):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size
        
        # Adding hidden layers based on depth (number of hidden_layers) and width
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.ReLU())
            in_features = width
        
        # Final output layer
        layers.append(nn.Linear(width, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Dataset loading function
def load_dataset(dataset_name, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset_name == 'mnist':
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        input_size = 28 * 28  # For MNIST, 28x28 images
        output_size = 10      # 10 classes
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        input_size = 32 * 32 * 3  # CIFAR10 images are 32x32 with 3 channels (RGB)
        output_size = 10          # 10 classes
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        input_size = 32 * 32 * 3  # CIFAR100 images are 32x32 with 3 channels (RGB)
        output_size = 100         # 100 classes
    elif dataset_name == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        trainset = datasets.ImageFolder(root='/raid/datasets/imagenet/train', transform=transform)
        input_size = 224 * 224 * 3  # ImageNet has larger images, 224x224 with 3 channels
        output_size = 1000          # 1000 classes
    else:
        raise ValueError("Dataset not supported! Choose from 'mnist', 'cifar10', 'cifar100', 'imagenet'.")

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return train_loader, input_size, output_size

# Train function with progress info and progress bars
def train(model, train_loader, device, time_limit=180):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    model.to(device)
    start_time = time.time()

    epoch = 0
    while True:  # Infinite loop, training will stop after 3 minutes
        epoch_loss = 0.0
        total_batches = len(train_loader)

        # Progress bar for the epoch
        with tqdm(total=total_batches, desc=f'Epoch {epoch + 1}', unit='batch') as pbar:
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Check if time limit has been exceeded
                if time.time() - start_time > time_limit:
                    print(f"Training stopped due to time limit ({time_limit} seconds)")
                    return
                
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs.view(inputs.size(0), -1))  # Flatten the inputs for MLP
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        
        # Print epoch progress
        avg_loss = epoch_loss / total_batches
        print(f"Epoch {epoch + 1} complete. Average Loss: {avg_loss:.4f}")
        epoch += 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a configurable MLP on MNIST, CIFAR10, CIFAR100, or ImageNet.')
    
    # Add command-line arguments for depth, width, batch size, and dataset selection
    parser.add_argument('--depth', type=int, default=3, help='Number of hidden layers (depth of the MLP)')
    parser.add_argument('--width', type=int, default=128, help='Number of neurons in each hidden layer (width)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'cifar100', 'imagenet'], default='mnist', help='Dataset to train on')
    
    args = parser.parse_args()

    # Hyperparameters from the command-line arguments
    depth = args.depth
    width = args.width
    batch_size = args.batch_size
    dataset_name = args.dataset

    # Load the dataset
    train_loader, input_size, output_size = load_dataset(dataset_name, batch_size)

    # Create the MLP model
    model = MLP(input_size, output_size, depth, width).to(device)

    # Print model summary
    summary(model, input_size=(1, input_size), device=device)
    
    # Train the model with a time limit of 3 minutes (180 seconds)
    train(model, train_loader, device, time_limit=180)