import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from torchsummary import summary
from tqdm import tqdm
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define dictionary for supported activation functions
activation_functions = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'prelu': nn.PReLU,
    'elu': nn.ELU,
    'selu': nn.SELU,
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'swish': nn.SiLU,  # Also known as Swish
    'softplus': nn.Softplus,
    'mish': nn.Mish
}

# Define your CNN architecture class
class CNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, architecture='pyramid', 
                 base_num_filters=32, filter_size=3, depth=4, 
                 use_pooling=True, use_dropout=True, dropout_rate=0.5, input_size=(128, 128),
                 use_skip=True, use_dilated=True, use_depthwise_separable=True, 
                 use_batch_norm=False, activation_function='relu'):
        super(CNN, self).__init__()

        layers = []
        in_channels = input_channels
        current_size = input_size  # Track the size of the input at each layer

        # Cap the maximum number of filters to avoid excessive growth
        max_filters = 2048

        # Choose activation function based on input
        activation_fn = activation_functions.get(activation_function, nn.ReLU)  # Default to ReLU if not specified

        # Set number of filters and kernel sizes based on architecture type
        if architecture == 'pyramid':
            num_filters = [min(base_num_filters * (2 ** i), max_filters) for i in range(depth)]
        elif architecture == 'reverse_pyramid':
            num_filters = [max(1, base_num_filters // (2 ** i)) for i in range(depth)]
        elif architecture == 'gradual':
            num_filters = [min(base_num_filters + i * 16, max_filters) for i in range(depth)]
        elif architecture == 'uniform':
            num_filters = [base_num_filters for _ in range(depth)]
        elif architecture == 'bottleneck':
            num_filters = [base_num_filters * 2 if i == 0 else base_num_filters // 2 for i in range(depth)]
        elif architecture == 'hourglass':
            num_filters = [int(base_num_filters * (2 if i < depth // 2 else 0.5)) for i in range(depth)]
        elif architecture == 'residual':
            num_filters = [base_num_filters for _ in range(depth)]
        elif architecture == 'dense':
            num_filters = [base_num_filters for _ in range(depth)]
        else:
            raise ValueError("Unsupported architecture type.")

        # Build convolutional layers
        for i in range(depth):
            print(f"Adding layer {i+1}/{depth}")
            print(f"in_channels: {in_channels}, out_channels: {num_filters[i]}")
            print(f"Current input size: {current_size}")

            out_channels = num_filters[i]
            kernel_size = filter_size
            stride = 1 if i < depth - 1 else 2  # Apply downsampling (stride=2) only in deeper layers or at end

            # Convolution layer: Can be depthwise separable or regular
            if use_depthwise_separable:
                layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=1, groups=in_channels))
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))  # Pointwise convolution
            else:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1))
            
            # Add batch normalization if enabled
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))

            layers.append(activation_fn())

            # Apply dilated convolution if selected
            if use_dilated:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=2, dilation=2))
                if use_batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(activation_fn())

            # Apply skip connections for residual architecture
            if use_skip and i > 0 and architecture == 'residual':
                skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
                layers.append(skip_connection)
                layers.append(activation_fn())

            # Apply dense connections for DenseNet-style architecture
            if architecture == 'dense' and i > 0:
                layers.append(nn.Conv2d(in_channels + num_filters[i - 1], out_channels, kernel_size=kernel_size, stride=stride, padding=1))

            # Print after adding each layer
            print(f"Added Conv2d layer with {in_channels} -> {out_channels} channels, kernel_size: {kernel_size}, stride: {stride}")
            
            # Update current size of the feature map
            current_size = ((current_size[0] - kernel_size + 2 * 1) // stride + 1, 
                            (current_size[1] - kernel_size + 2 * 1) // stride + 1)

            if current_size[0] < 1 or current_size[1] < 1:
                raise ValueError(f"Invalid architecture: output size ({current_size}) is too small.")

            # Apply pooling layer every 3 layers for downsampling
            if use_pooling and i % 3 == 0:  
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_size = (current_size[0] // 2, current_size[1] // 2)

                if current_size[0] < 1 or current_size[1] < 1:
                    raise ValueError(f"Invalid architecture: output size ({current_size}) is too small after pooling.")

            print(f"Updated input size: {current_size}")

            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Fully connected layers
        final_size = current_size[0] * current_size[1] * out_channels
        self.fc1 = nn.Linear(final_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Decide the final activation intelligently
        if num_classes == 1:
            if random.random() < 0.5:
                print("Chosen Sigmoid for binary classification")
                self.final_activation = nn.Sigmoid()  # Binary classification
            else:
                print("Chosen no activation for regression")
                self.final_activation = nn.Identity()  # Regression
        else:
            self.final_activation = nn.Softmax(dim=1)  # Multi-class classification

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.final_activation(x)
        return x

# Custom Dataset class for on-the-fly data generation
class OnTheFlyDataset(Dataset):
    def __init__(self, num_samples, input_size, channels, num_classes):
        self.num_samples = num_samples
        self.input_size = input_size
        self.channels = channels
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random input data and label on-the-fly
        input_data = torch.randn(self.channels, *self.input_size, dtype=torch.float32)
        label = torch.randint(0, self.num_classes, (1,)).item()  # Random label for classification
        return input_data, label

# Modify the data loading to use the new dataset
def get_data_loader(input_size, channels, num_classes, num_samples, batch_size):
    dataset = OnTheFlyDataset(num_samples=num_samples, input_size=input_size, channels=channels, num_classes=num_classes)
    # Reduced num_workers to 2 and added pin_memory=True to improve memory efficiency
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)  
    return data_loader

# Model training function with tqdm for progress tracking
def train_model(model, input_shape, num_classes, batch_size=128, learning_rate=0.001, num_epochs=5, dataset_size=10000, time_limit=60):
    model = model.to(device)

    # Create the data loader
    data_loader = get_data_loader(input_size=input_shape[1:], channels=input_shape[0], num_classes=num_classes, num_samples=dataset_size, batch_size=batch_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss() if num_classes > 1 else torch.nn.BCELoss()

    model.train()  # Set the model to training mode

    # Initialize timing variables if time limit is used
    start_time = time.time() if time_limit is not None else None

    print(f"Starting training for {num_epochs} epochs or until {time_limit} seconds is reached...")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_batches = len(data_loader)

        # Progress bar for each epoch
        with tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{num_epochs}", unit='batch', leave=False) as pbar:
            for batch_idx, (inputs, labels) in enumerate(data_loader):

                # If time limit is specified and exceeded, stop training
                if time_limit and time.time() - start_time > time_limit:
                    print(f"\nTraining stopped due to time limit of {time_limit} seconds.")
                    return
                
                # Move inputs and labels to the specified device (cuda or cpu)
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization step
                loss.backward()
                optimizer.step()

                # Accumulate loss for the epoch
                epoch_loss += loss.item()

                # Update the epoch progress bar
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)

        # Print loss for the epoch
        print(f"Epoch {epoch + 1}/{num_epochs} completed with loss: {epoch_loss / total_batches:.4f}")

        # Check if the time limit has been exceeded
        if time_limit and time.time() - start_time > time_limit:
            print(f"Training stopped after {epoch + 1} epochs due to time limit.")
            break

    print(f"Training completed after {num_epochs} epochs or time limit of {time_limit} seconds.")

# Main function using argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a configurable CNN with different architectures on a dummy dataset.')
    parser.add_argument('--depth', type=int, default=20, help='Depth of the CNN (number of convolutional layers)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--input_size', type=int, nargs=2, default=[128, 128], help='Input size as (height, width) tuple')
    parser.add_argument('--channels', type=int, default=3, help='Channels of input image')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes (output size)')
    parser.add_argument('--architecture', type=str, choices=['pyramid', 'reverse_pyramid', 'uniform', 'bottleneck', 'gradual', 'hourglass', 'residual', 'dense'], default='pyramid', help='Architecture type for the CNN')
    parser.add_argument('--num_samples', type=int, default=4096, help='Number of samples in the dummy dataset')
    parser.add_argument('--base_num_filters', type=int, default=64, help='Base number of filters for the CNN')
    parser.add_argument('--use_pooling', default=False, action='store_true', help='Use pooling layers')
    parser.add_argument('--use_dropout', default=False, action='store_true', help='Use dropout layers')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--use_skip', default=False, action='store_true', help='Use skip connections in CNN')
    parser.add_argument('--use_dilated', default=False, action='store_true', help='Use dilated convolutions in CNN')
    parser.add_argument('--use_depthwise_separable', default=False, action='store_true', help='Use depthwise separable convolutions in CNN')
    parser.add_argument('--use_batch_norm', default=False, action='store_true', help='Use batch normalization layers')
    parser.add_argument('--activation_function', type=str, choices=activation_functions.keys(), default='relu', help='Activation function to use in hidden layers')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--time_limit', type=int, default=60, help='Time limit for training in seconds')

    args = parser.parse_args()

    input_size_tuple = tuple(args.input_size)

    # Create the model
    model = CNN(input_channels=args.channels, num_classes=args.num_classes, architecture=args.architecture, 
                base_num_filters=args.base_num_filters, filter_size=3, 
                depth=args.depth, use_pooling=args.use_pooling, use_dropout=args.use_dropout, 
                dropout_rate=args.dropout_rate, input_size=input_size_tuple,
                use_skip=args.use_skip, use_dilated=args.use_dilated, use_depthwise_separable=args.use_depthwise_separable,
                use_batch_norm=args.use_batch_norm, activation_function=args.activation_function)

    # Move the model to the correct device (cuda or cpu)
    model = model.to(device)

    # Make sure that the summary is called with the model already moved to the right device
    summary(model, input_size=(args.channels, *input_size_tuple), device=device)

    # Train the model using the parameters from argparse
    train_model(model, input_shape=(args.channels, *input_size_tuple), 
                num_classes=args.num_classes, 
                batch_size=args.batch_size, 
                num_epochs=args.num_epochs, 
                dataset_size=args.num_samples, 
                time_limit=args.time_limit, 
                learning_rate=args.learning_rate)