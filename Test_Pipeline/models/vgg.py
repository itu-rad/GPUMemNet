import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm  # For progress bar
from torchsummary import summary  # For model summary
from contextlib import redirect_stdout

def train_vgg(batch_size, num_epochs=10, learning_rate=0.001, data_dir='/raid/datasets/imagenet', meta_cache_dir='/home/ehyo/.torch_cache/imagenet_meta'):
    # Set the TORCH_HOME environment variable to use the cache directory for metadata
    os.environ['TORCH_HOME'] = meta_cache_dir

    # Define transformations for the training and validation datasets
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Use ImageFolder instead of ImageNet to bypass the internal metadata saving issue
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)

    # Create DataLoader for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load the VGG model
    model = models.vgg16(pretrained=False)  # You can change to other VGG variants (e.g., vgg11, vgg19)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1000)  # Adjust the final layer for ImageNet classes

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Save model summary to a file
    summary_path = f'vgg_{batch_size}.model'
    with open(summary_path, 'w') as f:
        # Redirect the stdout to capture the summary output
        with redirect_stdout(f):
            summary(model, (3, 224, 224), batch_size=batch_size, device=str(device))
    print(f"Model summary saved to {summary_path}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Use tqdm for the training progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}]', unit='batch')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

        # Validation (optional)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Loss: {val_loss / len(val_loader)}, Accuracy: {100 * correct / total:.2f}%")

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VGG on ImageNet.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--data_dir', type=str, default='/raid/datasets/imagenet', help='Directory for ImageNet dataset.')
    parser.add_argument('--meta_cache_dir', type=str, default='/home/ehyo/.torch_cache/imagenet_meta', help='Directory for storing metadata cache.')
    
    args = parser.parse_args()

    train_vgg(batch_size=args.batch_size, num_epochs=args.num_epochs, learning_rate=args.learning_rate, data_dir=args.data_dir, meta_cache_dir=args.meta_cache_dir)
