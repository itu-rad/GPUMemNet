import pytorch_lightning as pl
import torch.nn as nn
import torch

class RandomMLP(pl.LightningModule):
    def __init__(self, input_size, output_size, depth, max_neurons, min_neurons, learning_rate):
        """
        Initializes a random MLP model with a decreasing number of neurons as layers go deeper.
        
        Parameters:
        - input_size (int): Number of input features
        - output_size (int): Number of output classes (binary classification)
        - max_layers (int): Maximum number of hidden layers
        - max_neurons (int): Number of neurons in the first hidden layer
        - min_neurons (int): Minimum number of neurons in the last hidden layer
        - learning_rate (float): Learning rate for training
        """
        super(RandomMLP, self).__init__()
        self.learning_rate = learning_rate
        layers = []
        current_size = input_size
        # print("number of layers: ", depth)
        # Decreasing the number of neurons for each successive layer
        layer_sizes = [int(max_neurons * (0.5 ** i)) for i in range(depth)]
        layer_sizes = [max(min_neurons, size) for size in layer_sizes]  # Ensure each layer has at least min_neurons

        # Create each layer with BatchNorm and ReLU
        for next_size in layer_sizes:
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.BatchNorm1d(next_size))
            layers.append(nn.ReLU(inplace=True))
            current_size = next_size

        # Output layer (no batch norm or ReLU for the output layer)
        layers.append(nn.Linear(current_size, output_size))
        
        self.model = nn.Sequential(*layers)
        self.apply(self._initialize_weights)
        
    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)  # Xavier uniform initialization
            if module.bias is not None:
                nn.init.zeros_(module.bias) 

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class EnsembleModel(pl.LightningModule):
    def __init__(self, model_list, input_size, output_size, max_neurons, min_neurons, learning_rate):
        """
        Initializes an ensemble of random MLP models.
        
        Parameters:
        - model_list (list): a list of depth for mlps
        - input_size (int): Number of input features
        - output_size (int): Number of output classes (binary classification)
        - max_layers (int): Maximum number of hidden layers per MLP
        - max_neurons (int): Maximum number of neurons in the first hidden layer of each MLP
        - min_neurons (int): Minimum number of neurons in the last hidden layer of each MLP
        - learning_rate (float): Learning rate for training
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(
            [RandomMLP(input_size, output_size, depth, max_neurons, min_neurons, learning_rate) for depth in model_list]
        )
        self.learning_rate = learning_rate


    def forward(self, x):
        # Average the predictions from all models
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y.long())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y.long())
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
