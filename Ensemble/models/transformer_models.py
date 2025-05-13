import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils import read_yaml
# from dataloaders4transformer import transformer_data4transformer
from .mlp_models import EnsembleModel

class TransformerClassifier(pl.LightningModule):
    def __init__(self, num_features, num_classes, d_model, nhead, num_layers, dim_feedforward, dropout, max_seq_len, extra_fetures_num):
        super(TransformerClassifier, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.BatchNorm1d(max_seq_len),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(max_seq_len),
            nn.ReLU(),
        )
        self.positional_encoding = nn.Parameter(self._get_positional_encoding(max_seq_len, d_model), requires_grad=False)
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        # self.fc_out = nn.Sequential(
        #     nn.Linear(d_model + extra_fetures_num, d_model),
        #     nn.BatchNorm1d(d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, num_classes),
        # )
        self.fc_out = EnsembleModel(model_list=[1,2,3,4,5,6,7,1,2,3,4,5,6], input_size=d_model + extra_fetures_num, 
                                         output_size=num_classes, max_neurons=8, min_neurons=4, learning_rate=0.001)

    def _get_positional_encoding(self, max_seq_len, d_model):
        pos_encoding = torch.zeros(max_seq_len, d_model)
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pos_encoding[:, 0::2] = torch.sin(positions * div_term)
        pos_encoding[:, 1::2] = torch.cos(positions * div_term)
        return pos_encoding.unsqueeze(0)

    def forward(self, x, batch_size_feature):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        x = self.transformer_encoder(x)
        x = x.sum(dim=1)
        batch_size_feature = batch_size_feature.unsqueeze(1) if batch_size_feature.dim() == 1 else batch_size_feature
        x = torch.cat((x, batch_size_feature), dim=1)
        return self.fc_out(x)

class TransformerEnsemble(pl.LightningModule):
    def __init__(self, model_configs, num_features, num_classes, learning_rate, max_seq_len, extra_fetures_num):
        """
        Initializes an ensemble of TransformerClassifier models.
        
        Parameters:
        - model_configs (list of dict): List of configurations for each Transformer model in the ensemble
        - num_features (int): Number of input features
        - num_classes (int): Number of output classes
        - learning_rate (float): Learning rate for training
        """
        super(TransformerEnsemble, self).__init__()
        self.models = nn.ModuleList([
            TransformerClassifier(
                num_features=num_features,
                num_classes=num_classes,
                d_model=config['d_model'],
                nhead=config['nhead'],
                num_layers=config['num_layers'],
                dim_feedforward=config['dim_feedforward'],
                dropout=config['dropout'],
                max_seq_len=max_seq_len,
                extra_fetures_num=extra_fetures_num
            )
            for config in model_configs
        ])
        self.learning_rate = learning_rate

    def forward(self, x, batch_size_feature):
        # Average the predictions from all Transformer models in the ensemble
        outputs = [model(x, batch_size_feature) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

    def training_step(self, batch, batch_idx):
        x, batch_size_feature, y = batch
        logits = self(x, batch_size_feature)
        loss = nn.CrossEntropyLoss()(logits, y.long())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, batch_size_feature, y = batch
        logits = self(x, batch_size_feature)
        loss = nn.CrossEntropyLoss()(logits, y.long())
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
# if __name__ == "__main__":
#     config = read_yaml("config.yaml")
#     transformers_architectures = read_yaml("transformers_architectures.yaml")['model_configs']

#     train_dataloader, val_dataloader, test_dataloader, data_related_info = transformer_data4transformer(config)
#     x, z, y = next(iter(train_dataloader))
#     print('inputs shape: ', x.shape, z.shape)    
#     extra_features_num = z.shape[1]
#     num_features = x.shape[-1]
#     classifier_model = TransformerEnsemble(model_configs=transformers_architectures, num_features=num_features, 
#                                     num_classes=data_related_info["class_counts"], learning_rate=config["learning_rate"], 
#                                     max_seq_len=data_related_info["max_seq_len"], 
#                                     extra_fetures_num=extra_features_num)
    
#     out = classifier_model(x, z)
#     print('output shape: ', out.shape)    