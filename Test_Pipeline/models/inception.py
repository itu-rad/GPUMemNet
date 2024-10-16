import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from tqdm import tqdm
import argparse
import io  # Import io for capturing the summary output
from contextlib import redirect_stdout

def train_inception(batch_size, num_epochs=10, learning_rate=0.001, data_dir='/raid/datasets/imagenet'):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(299),  # InceptionV3 requires 299x299 input size
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the InceptionV3 model
    model = torchvision.models.inception_v3(weights=None, aux_logits=False, num_classes=1000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Capture and save model summary
    summary_path = f'inceptionv3_{batch_size}.model'
    with open(summary_path, 'w') as f:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            summary(model, input_size=(3, 299, 299), device=str(device))
        f.write(buffer.getvalue())
    print(f"Model summary saved to {summary_path}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}]', unit='batch')

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

        # Validation
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

# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train InceptionV3 on ImageNet.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--data_dir', type=str, default='/raid/datasets/imagenet', help='Directory for ImageNet dataset.')
    args = parser.parse_args()

    train_inception(batch_size=args.batch_size, num_epochs=args.num_epochs, learning_rate=args.learning_rate, data_dir=args.data_dir)
