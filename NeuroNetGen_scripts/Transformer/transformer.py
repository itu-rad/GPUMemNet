import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, TensorDataset
from modelsummary import summary  # Import modelsummary

import time

class TransformerModel(nn.Module):
    def __init__(self, input_size, embedding_size, num_layers, num_heads, ff_hidden_size, dropout_rate, seq_length, output_size, architecture):
        super(TransformerModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_size)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, embedding_size, dtype=torch.float32))

        # Define different architectures
        if architecture == 'standard':
            self.model = self._build_standard_transformer(embedding_size, num_layers, num_heads, ff_hidden_size, dropout_rate)
        elif architecture == 'decoder_only':
            self.model = self._build_decoder_only_transformer(embedding_size, num_layers, num_heads, ff_hidden_size, dropout_rate)
        elif architecture == 'hybrid':
            self.model = self._build_hybrid_transformer(embedding_size, num_layers, num_heads, ff_hidden_size, dropout_rate)
        else:
            raise ValueError("Unknown architecture type")

        # Output layer
        self.fc_out = nn.Linear(embedding_size, output_size)

    def _build_standard_transformer(self, embedding_size, num_layers, num_heads, ff_hidden_size, dropout_rate):
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, dim_feedforward=ff_hidden_size, dropout=dropout_rate, batch_first=True)
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _build_decoder_only_transformer(self, embedding_size, num_layers, num_heads, ff_hidden_size, dropout_rate):
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=num_heads, dim_feedforward=ff_hidden_size, dropout=dropout_rate, batch_first=True)
        return nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def _build_hybrid_transformer(self, embedding_size, num_layers, num_heads, ff_hidden_size, dropout_rate):
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, dim_feedforward=ff_hidden_size, dropout=dropout_rate, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=num_heads, dim_feedforward=ff_hidden_size, dropout=dropout_rate, batch_first=True)
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers // 2)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers // 2)
        return nn.ModuleList([encoder, decoder])

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding  # Add positional encoding
        if isinstance(self.model, nn.ModuleList):  # Hybrid model with encoder-decoder
            encoder, decoder = self.model
            memory = encoder(x)
            x = decoder(x, memory)
        else:  # Standard or decoder-only model
            x = self.model(x)

        # Pooling (take the mean of sequence dimension)
        x = x.mean(dim=1)

        output = self.fc_out(x)

        return output  # Ensure the output is always returned

# Function to generate Transformer model based on input arguments
def generate_transformer(args):
    model = TransformerModel(input_size=args.input_size,
                             embedding_size=args.embedding_size,
                             num_layers=args.num_layers,
                             num_heads=args.num_heads,
                             ff_hidden_size=args.ff_hidden_size,
                             dropout_rate=args.dropout_rate,
                             seq_length=args.seq_length,
                             output_size=args.output_size,
                             architecture=args.architecture)
    return model

# Optimized dummy dataset generator (classification)
class OptimizedDummyDataset(TensorDataset):
    def __init__(self, input_size, seq_length, num_samples, num_classes):
        self.input_size = input_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a single sample on-the-fly to save memory
        X = torch.randint(0, self.input_size, (self.seq_length,), dtype=torch.long)
        y = torch.randint(0, self.num_classes, (1,), dtype=torch.long).item()
        return X, y

def get_data_loader(input_size, seq_length, num_samples, num_classes, batch_size):
    dataset = OptimizedDummyDataset(input_size=input_size, seq_length=seq_length, num_samples=num_samples, num_classes=num_classes)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# Training function
def train_model(model, dataloader, criterion, optimizer, device, epochs=10, time_limit=60):
    model.to(device)
    model.train()

    start_time = time.time()  # Start timing the overall training process

    for epoch in range(epochs):
        # Check if time limit is exceeded at the beginning of each epoch
        if time_limit and time.time() - start_time > time_limit:
            print(f"Time limit exceeded at epoch {epoch + 1}")
            break  # Exit the loop if the time limit has been exceeded

        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Check if time limit is exceeded during batch processing
            if time_limit and time.time() - start_time > time_limit:
                print(f"Time limit exceeded during epoch {epoch + 1} after batch {batch[0]}")  # Optional: indicate which batch
                break  # Exit the loop if the time limit has been exceeded

        # Calculate accuracy after the epoch
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and train a Transformer model with user-specified parameters.")
    
    # Define the command-line arguments
    parser.add_argument('--input_size', type=int, default=100000, help='Size of the input vocabulary or feature space')
    parser.add_argument('--embedding_size', type=int, default=512, help='Size of the embedding vector')
    parser.add_argument('--num_layers', type=int, default=10, help='Number of transformer layers (depth)')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ff_hidden_size', type=int, default=2048, help='Size of the feed-forward hidden layer')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--seq_length', type=int, default=256, help='Length of the input sequence')
    parser.add_argument('--output_size', type=int, default=10, help='Output size (number of classes or regression output size)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples in the dummy dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--architecture', type=str, default='standard', choices=['standard', 'decoder_only', 'hybrid'], help='Choose the transformer architecture')

    # Parse the arguments
    args = parser.parse_args()

    # Generate Transformer model
    model = generate_transformer(args)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create a mock input tensor for the summary
    mock_input = torch.randint(0, args.input_size, (args.batch_size, args.seq_length), dtype=torch.long).to(device)

    # Check model compatibility with the mock input
    try:
        model(mock_input)  # Test with the mock input
        print("Dummy input works with the model.")
    except Exception as e:
        print(f"Error during dummy input check: {e}")

    # Call summary using modelsummary to show output
    try:
        summary(model, mock_input, show_input=True, show_hierarchical=True)  # Provide correct input size
    except Exception as e:
        print(f"Error during summary: {e}")

    # Create DataLoader for batching
    dataloader = get_data_loader(args.input_size, args.seq_length, args.num_samples, args.output_size, args.batch_size)  
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, device, epochs=args.epochs, time_limit=60)  # Limit training to 1 minute
