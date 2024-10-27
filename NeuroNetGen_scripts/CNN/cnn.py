import torch
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torchinfo  import summary
import time
import math
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define dictionary for activation functions
activation_functions = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'prelu': nn.PReLU,
    'elu': nn.ELU,
    'selu': nn.SELU,
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'swish': nn.SiLU,
    'softplus': nn.Softplus,
    'mish': nn.Mish
}

# Basic ConvBlock for reuse
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batch_norm=False, use_dropout=False, dropout_rate=0.5, activation='relu'):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(activation_functions[activation]())
        
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# Residual Block with skip connections (Addition)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_batch_norm=False, use_dropout=False, dropout_rate=0.5, activation='relu'):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride, use_batch_norm=use_batch_norm, use_dropout=use_dropout, dropout_rate=dropout_rate, activation=activation)
        self.conv2 = ConvBlock(out_channels, out_channels, stride=1, use_batch_norm=use_batch_norm, use_dropout=use_dropout, dropout_rate=dropout_rate, activation=activation)
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual

# Dense Block with concatenation
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, use_batch_norm=False, use_dropout=False, dropout_rate=0.5, activation='relu'):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = ConvBlock(in_channels + i * growth_rate, growth_rate, use_batch_norm=use_batch_norm, use_dropout=use_dropout, dropout_rate=dropout_rate, activation=activation)
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))  # Concatenate along channel dimension
            features.append(new_feature)
        return torch.cat(features, 1)  # Return the concatenated output

# Transition block for DenseNet to downsample
class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=False):
        super(TransitionBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_batch_norm=use_batch_norm)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        if x.size(2) > 1 and x.size(3) > 1:  # Ensure spatial dimensions are > 1 before pooling
            x = self.pool(x)
        return x

# Block-based CNN with architectural variations
class CNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, architecture='pyramid', base_num_filters=32, depth=4, growth_rate=32, use_batch_norm=False, use_dropout=False, dropout_rate=0.5, use_pooling=False, use_dilated=False, activation='relu', input_size=(128, 128)):
        super(CNN, self).__init__()

        self.blocks = nn.ModuleList()
        in_channels = input_channels
        max_filters = 2048  # Cap filters to avoid excessive size
        current_size = input_size  # Keep track of the spatial dimensions

        num_residual_blocks = depth // 2
        dense_layers_count = depth // 4

        random_boolean = np.random.rand() > 0.5

        target_size = 7
        # Calculate the total number of times we need to pool to go from input_size to target_size
        total_poolings_needed = max(1, int(math.log2(input_size[0] // target_size)))  # Assuming input size is square

        print(random_boolean)
        if random_boolean:
            # Calculate pooling frequency based on the depth of the network
            pooling_frequency = max(1, math.ceil(depth / total_poolings_needed))
            print(pooling_frequency)
        else:
            pooling_frequency = 5
            print(pooling_frequency)
        

        # Handle Residual Blocks (ResNet style)
        if architecture == 'residual':
            growth_factor = (max_filters / base_num_filters) ** (1 / depth)  # Residual growth factor based on depth
            for i in range(num_residual_blocks):
                out_channels = min(int(base_num_filters * (growth_factor ** i)), max_filters)
                stride = 2 if i % 2 == 1 and current_size[0] > 7 else 1
                block = ResidualBlock(in_channels, out_channels, stride=stride, use_batch_norm=use_batch_norm, use_dropout=use_dropout, dropout_rate=dropout_rate, activation=activation)
                self.blocks.append(block)
                in_channels = out_channels

                # Update spatial size if stride=2 (downsampling)
                if stride == 2:
                    current_size = (current_size[0] // 2, current_size[1] // 2)

                if current_size[0] <= 7:  # Stop further shrinking of spatial dimensions
                    break

        # Handle Dense Blocks (DenseNet style)
        elif architecture == 'dense':
            growth_factor = (max_filters / base_num_filters) ** (1 / (2 * depth))  # Dense architecture grows more gradually
            for i in range(dense_layers_count):
                block = DenseBlock(in_channels, growth_rate, num_layers=4, use_batch_norm=use_batch_norm, use_dropout=use_dropout, dropout_rate=dropout_rate, activation=activation)
                self.blocks.append(block)
                in_channels += growth_rate * 4


                if current_size[0] >= 14:
                    current_size = (current_size[0] // 2, current_size[1] // 2)

                if i % pooling_frequency == 0 and i != dense_layers_count - 1:
                    out_channels = min(int(in_channels * growth_factor), max_filters)  # Apply growth factor to transition layers
                    self.blocks.append(TransitionBlock(in_channels, in_channels // 2, use_batch_norm=use_batch_norm))
                    in_channels = in_channels // 2

                continue

        # Handle Other Architectures
        else:
            # Set target size (e.g., reducing to at least 7x7 before fully connected layers)
            target_size = 7
            
            # Calculate the total number of times we need to pool to go from input_size to target_size
            total_poolings_needed = max(1, int(math.log2(input_size[0] // target_size)))  # Assuming input size is square


            random_boolean = np.random.rand() > 0.5

            print("Decision: ", random_boolean)

            if random_boolean:
            # Calculate pooling frequency based on the depth of the network
                # pooling_frequency = max(1, depth // 10)  # Example: Pool every depth//10 layers
                pooling_interval = max(1, math.ceil(depth / total_poolings_needed))
                print("pooling interval (other archs than res and dense)", pooling_interval)
            else:
                pooling_interval = 5
                print("pooling interval (other archs than res and dense)", pooling_interval)


            
            # logic for hourglass architecture
            # Hourglass architecture requires careful management of filter growth and shrinking
            contracting_phase = depth // 2  # Contracting phase (reduce spatial size, increase channels)
            expanding_phase = depth - contracting_phase  # Expanding phase (increase spatial size, reduce channels)

            

            for i in range(depth):
                if architecture == 'pyramid':
                    # Pyramid architecture grows filters fast; growth factor calculated to exponentially increase filters
                    growth_factor = (max_filters / base_num_filters) ** (1 / depth)
                    out_channels = min(int(base_num_filters * (growth_factor ** i)), max_filters)

                elif architecture == 'reverse_pyramid':
                    # Reverse Pyramid architecture decreases filters with depth
                    growth_factor =  (max_filters/ base_num_filters) ** (1 / depth)  # Inverse growth factor
                    out_channels = max(int(base_num_filters / (growth_factor ** i)), 1)  # Ensure filters decrease

                elif architecture == 'bottleneck':
                    # Bottleneck starts with a sudden increase, then grows more steadily
                    if depth <= 2:
                        growth_factor = 2  # Rapid growth in shallow networks
                    else:
                        growth_factor = (max_filters / base_num_filters) ** (1 / (depth - 1))

                    if i == 0:
                        out_channels = base_num_filters * 2  # Bottleneck at the first layer
                    else:
                        out_channels = min(int(base_num_filters * (growth_factor ** (i - 1))), max_filters)

                elif architecture == 'gradual':
                    # Gradual architecture grows filters slower; growth factor calculated for sub-exponential increase
                    growth_factor = (max_filters / base_num_filters) ** (1 / (2 * depth))  # Half the growth rate of pyramid
                    out_channels = min(int(base_num_filters * (growth_factor ** i)), max_filters)

                elif architecture == 'hourglass':
                    if i < contracting_phase:
                        growth_factor = (max_filters / base_num_filters) ** (1 / contracting_phase)
                        # Contracting Phase (increase filters, reduce spatial size)
                        out_channels = min(int(base_num_filters * (growth_factor ** i)), max_filters)
                        print("expand, out channels: ", out_channels)
                    else:
                        # Expanding Phase (reduce filters, increase spatial size)
                        # Shrinking factor to reverse the growth
                        shrink_factor = (max_filters / base_num_filters) ** (1 / expanding_phase)
                        out_channels = max(int(base_num_filters * (shrink_factor ** (depth - i - 1))), base_num_filters)
                        print("shrink, out channels: ", out_channels)

                        
                elif architecture == 'uniform':
                    # Uniform architecture keeps the filter size constant throughout
                    growth_factor = 1  # No growth
                    out_channels = base_num_filters

                else:
                    raise ValueError(f"Unsupported architecture: {architecture}")

                block = ConvBlock(in_channels, out_channels, use_batch_norm=use_batch_norm, use_dropout=use_dropout, dropout_rate=dropout_rate, activation=activation)
                self.blocks.append(block)
                in_channels = out_channels

                # Optional pooling
                # Apply pooling every 5 layers or based on depth, ensuring the current size shrinks.
                if use_pooling and (i % pooling_interval == 0) and i != 0 and current_size[0] > 7:
                    self.blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
                    current_size = (current_size[0] // 2, current_size[1] // 2)

        # Global average pooling to reduce the feature map to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(in_channels, num_classes)

        # Final layer activation logic
        self.final_activation = nn.Softmax(dim=1) if num_classes > 1 else nn.Sigmoid()

    def forward_conv(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.final_activation(x)

# Dataset and training code remain the same
class OnTheFlyDataset(Dataset):
    def __init__(self, num_samples, input_size, channels, num_classes):
        self.num_samples = num_samples
        self.input_size = input_size
        self.channels = channels
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_data = torch.randn(self.channels, *self.input_size, dtype=torch.float32)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return input_data, label

def get_data_loader(input_size, channels, num_classes, num_samples, batch_size):
    dataset = OnTheFlyDataset(num_samples=num_samples, input_size=input_size, channels=channels, num_classes=num_classes)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return data_loader

def train_model(model, input_shape, num_classes, batch_size=128, learning_rate=0.001, num_epochs=5, dataset_size=10000, time_limit=60):
    model = model.to(device)
    data_loader = get_data_loader(input_size=input_shape[1:], channels=input_shape[0], num_classes=num_classes, num_samples=dataset_size, batch_size=batch_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss() if num_classes > 1 else torch.nn.BCELoss()

    model.train()
    start_time = time.time() if time_limit is not None else None

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_batches = len(data_loader)

        for batch_idx, (inputs, labels) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            if time_limit and time.time() - start_time > time_limit:
                print(f"Time limit exceeded at epoch {epoch + 1}")
                return
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs} completed with loss: {epoch_loss / total_batches:.4f}")

# Main function using argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a configurable CNN with different architectures on a dummy dataset.')
    parser.add_argument('--depth', type=int, default=10, help='Depth of the CNN (number of convolutional layers)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--input_size', type=int, nargs=2, default=[112, 112], help='Input size as (height, width) tuple')
    parser.add_argument('--channels', type=int, default=3, help='Channels of input image')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes (output size)')
    parser.add_argument('--architecture', type=str, choices=['pyramid', 'reverse_pyramid', 'uniform', 'bottleneck', 'gradual', 'hourglass', 'residual', 'dense'], default='pyramid', help='Architecture type for the CNN')
    parser.add_argument('--num_samples', type=int, default=4096, help='Number of samples in the dummy dataset')
    parser.add_argument('--base_num_filters', type=int, default=64, help='Base number of filters for the CNN')
    parser.add_argument('--use_pooling', default=False, action='store_true', help='Use pooling layers')
    parser.add_argument('--use_dropout', default=False, action='store_true', help='Use dropout layers')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--use_dilated', default=False, action='store_true', help='Use dilated convolutions in CNN')
    parser.add_argument('--use_batch_norm', default=False, action='store_true', help='Use batch normalization layers')
    parser.add_argument('--activation_function', type=str, choices=activation_functions.keys(), default='relu', help='Activation function to use in hidden layers')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--time_limit', type=int, default=60, help='Time limit for training in seconds')

    args = parser.parse_args()

    input_size_tuple = tuple(args.input_size)

    # Create the model
    model = CNN(input_channels=args.channels, num_classes=args.num_classes, architecture=args.architecture, 
                base_num_filters=args.base_num_filters, depth=args.depth, use_batch_norm=args.use_batch_norm, 
                use_dropout=args.use_dropout, dropout_rate=args.dropout_rate, use_pooling=args.use_pooling, 
                use_dilated=args.use_dilated, activation=args.activation_function, input_size=input_size_tuple)

    # Move the model to the correct device (cuda or cpu)
    model = model.to(device)

    print(args.input_size)
    # Display the model summary
    summary(model, input_size=(args.batch_size, args.channels, args.input_size[0], args.input_size[1]))

    # Train the model using the parameters from argparse
    train_model(model, input_shape=(args.channels, *input_size_tuple), 
                num_classes=args.num_classes, 
                batch_size=args.batch_size, 
                num_epochs=args.num_epochs, 
                dataset_size=args.num_samples, 
                time_limit=args.time_limit, 
                learning_rate=args.learning_rate)
