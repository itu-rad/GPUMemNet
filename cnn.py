import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Pool, cpu_count
import time
from torchsummary import summary
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define your CNN architecture class
class CNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, architecture='pyramid', 
                 base_num_filters=32, filter_size=3, depth=4, 
                 use_pooling=True, use_dropout=True, dropout_rate=0.5, input_size=(128, 128),
                 use_skip=True, use_dilated=True, use_depthwise_separable=True):
        super(CNN, self).__init__()

        layers = []
        in_channels = input_channels
        current_size = input_size  # Track the size of the input at each layer

        # Set number of filters and kernel sizes based on architecture type
        if architecture == 'pyramid':
            num_filters = [base_num_filters * (2 ** i) for i in range(depth)]
        elif architecture == 'reverse_pyramid':
            num_filters = [base_num_filters // (2 ** i) if base_num_filters // (2 ** i) > 0 else 1 for i in range(depth)]
        elif architecture == 'gradual':
            num_filters = [base_num_filters + i * 16 for i in range(depth)]
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
            out_channels = num_filters[i]
            kernel_size = filter_size
            stride = 1 if i < depth - 1 else 2  # Apply downsampling (stride=2) only in deeper layers or at end

            # Convolution layer: Can be depthwise separable or regular
            if use_depthwise_separable:
                layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=1, groups=in_channels))
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))  # Pointwise convolution
            else:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1))
            
            layers.append(nn.ReLU())

            # Apply dilated convolution if selected
            if use_dilated:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=2, dilation=2))
                layers.append(nn.ReLU())

            # Apply skip connections for residual architecture
            if use_skip and i > 0 and architecture == 'residual':
                skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
                layers.append(skip_connection)
                layers.append(nn.ReLU())

            # Apply dense connections for DenseNet-style architecture
            if architecture == 'dense' and i > 0:
                layers.append(nn.Conv2d(in_channels + num_filters[i - 1], out_channels, kernel_size=kernel_size, stride=stride, padding=1))

            # Update current size of the feature map
            current_size = ((current_size[0] - kernel_size + 2 * 1) // stride + 1, 
                            (current_size[1] - kernel_size + 2 * 1) // stride + 1)

            if current_size[0] < 1 or current_size[1] < 1:
                raise ValueError(f"Invalid architecture: output size ({current_size}) is too small.")

            # Apply pooling layer only when necessary
            if use_pooling and i % 3 == 0:  # Apply pooling every 3 layers for downsampling
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_size = (current_size[0] // 2, current_size[1] // 2)

                # Ensure that pooling doesn't shrink the spatial dimensions to zero
                if current_size[0] < 1 or current_size[1] < 1:
                    raise ValueError(f"Invalid architecture: output size ({current_size}) is too small after pooling.")

            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Fully connected layers
        final_size = current_size[0] * current_size[1] * out_channels
        self.fc1 = nn.Linear(final_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Label generation function
def generate_labels(output_size, num_samples):
    return torch.randint(0, output_size, (num_samples,))

# Dataset generation function
def generate_dummy_dataset(input_size, output_size, num_samples, batch_size):
    inputs = torch.randn(num_samples, *input_size)
    
    # Use multiprocessing to speed up label generation
    num_cores = min(cpu_count(), num_samples)
    chunk_size = num_samples // num_cores
    
    pool = Pool(num_cores)
    labels_chunks = pool.starmap(generate_labels, [(output_size, chunk_size) for _ in range(num_cores)])
    
    remainder = num_samples % num_cores
    if remainder:
        labels_chunks.append(generate_labels(output_size, remainder))
    
    labels = torch.cat(labels_chunks)
    pool.close()
    pool.join()
    
    dataset = TensorDataset(inputs, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# Model training function with tqdm for progress tracking
def train_model(model, input_shape, num_classes, batch_size=128, learning_rate=0.001, train_time=60, dataset_size=10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    data_loader = generate_dummy_dataset(input_shape, num_classes, dataset_size, batch_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()

    start_time = time.time()
    pbar = tqdm(data_loader, desc="Training Progress", total=len(data_loader))

    for batch_data, batch_labels in pbar:
        if time.time() - start_time > train_time:
            break
        
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        pbar.set_postfix({'Loss': loss.item()})

    print(f"Training stopped after {train_time} seconds.")

# Main function using argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a configurable CNN with different architectures on a dummy dataset.')
    parser.add_argument('--depth', type=int, default=20, help='Depth of the CNN (number of convolutional layers)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--input_size', type=int, default=334, help='Input size (spatial dimensions of input image)')
    parser.add_argument('--channels', type=int, default=3, help='(Channels of input image)')
    parser.add_argument('--num_classes', type=int, default=100, required=True, help='Number of classes (output size)')
    parser.add_argument('--architecture', type=str, choices=['pyramid', 'reverse_pyramid', 'uniform', 'bottleneck', 'gradual', 'hourglass', 'residual', 'dense'], default='pyramid', help='Architecture type for the CNN')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples in the dummy dataset')
    parser.add_argument('--base_num_filters', type=int, default=256, help='Base number of filters for the CNN')
    parser.add_argument('--use_pooling', default=True, action='store_true', help='Use pooling layers')
    parser.add_argument('--use_dropout', default=False, action='store_true', help='Use dropout layers')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--use_skip', default=False, action='store_true', help='Use skip connections in CNN')
    parser.add_argument('--use_dilated', default=False, action='store_true', help='Use dilated convolutions in CNN')
    parser.add_argument('--use_depthwise_separable', default=False, action='store_true', help='Use depthwise separable convolutions in CNN')
    
    args = parser.parse_args()

    # Create the model
    model = CNN(input_channels=args.channels, num_classes=args.num_classes, architecture=args.architecture, 
                base_num_filters=args.base_num_filters, filter_size=3, 
                depth=args.depth, use_pooling=args.use_pooling, use_dropout=args.use_dropout, 
                dropout_rate=args.dropout_rate, input_size=(args.input_size, args.input_size),
                use_skip=args.use_skip, use_dilated=args.use_dilated, use_depthwise_separable=args.use_depthwise_separable)

    summary(model, input_size=(args.channels, args.input_size, args.input_size), device=device)

    train_model(model, (args.channels, args.input_size, args.input_size), args.num_classes, args.batch_size, train_time=120, dataset_size=args.num_samples)
