import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, TensorDataset

class TransformerModel(nn.Module):
    def __init__(self, input_size, embedding_size, num_layers, num_heads, ff_hidden_size, dropout_rate, seq_length, output_size, architecture):
        super(TransformerModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_size)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, embedding_size))

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
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, dim_feedforward=ff_hidden_size, dropout=dropout_rate)
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _build_decoder_only_transformer(self, embedding_size, num_layers, num_heads, ff_hidden_size, dropout_rate):
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=num_heads, dim_feedforward=ff_hidden_size, dropout=dropout_rate)
        return nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def _build_hybrid_transformer(self, embedding_size, num_layers, num_heads, ff_hidden_size, dropout_rate):
        # A hybrid model could include encoder-decoder connections, residual connections, or variations in depth
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, dim_feedforward=ff_hidden_size, dropout=dropout_rate)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=num_heads, dim_feedforward=ff_hidden_size, dropout=dropout_rate)
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers // 2)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers // 2)
        return nn.ModuleList([encoder, decoder])

    def forward(self, x):
        # Add embedding and positional encoding
        x = self.embedding(x) + self.positional_encoding

        # Forward pass through the selected transformer architecture
        if isinstance(self.model, nn.ModuleList):  # Hybrid model with encoder-decoder
            encoder, decoder = self.model
            memory = encoder(x)
            x = decoder(x, memory)
        else:  # Standard or decoder-only model
            x = self.model(x)

        # Pooling (take the mean of sequence dimension)
        x = x.mean(dim=1)

        # Output layer
        output = self.fc_out(x)

        return output

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

# Dummy dataset generator (classification)
def generate_dummy_data(input_size, seq_length, num_samples, num_classes):
    X = torch.randint(0, input_size, (num_samples, seq_length))  # Random sequences
    y = torch.randint(0, num_classes, (num_samples,))            # Random class labels
    return X, y

# Training function
def train_model(model, dataloader, criterion, optimizer, device, epochs=10):
    model.to(device)
    model.train()

    for epoch in range(epochs):
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
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and train a Transformer model with user-specified parameters.")
    
    # Define the command-line arguments
    parser.add_argument('--input_size', type=int, default=10000, help='Size of the input vocabulary or feature space')
    parser.add_argument('--embedding_size', type=int, default=512, help='Size of the embedding vector')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers (depth)')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ff_hidden_size', type=int, default=2048, help='Size of the feed-forward hidden layer')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--seq_length', type=int, default=128, help='Length of the input sequence')
    parser.add_argument('--output_size', type=int, default=10, help='Output size (number of classes or regression output size)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples in the dummy dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--architecture', type=str, default='standard', choices=['standard', 'decoder_only', 'hybrid'], help='Choose the transformer architecture')

    # Parse the arguments
    args = parser.parse_args()

    # Generate Transformer model
    model = generate_transformer(args)

    # Generate dummy dataset for classification (X: input sequences, y: class labels)
    X, y = generate_dummy_data(args.input_size, args.seq_length, args.num_samples, args.output_size)

    # Create DataLoader for batching
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Specify the device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    train_model(model, dataloader, criterion, optimizer, device, epochs=args.epochs)