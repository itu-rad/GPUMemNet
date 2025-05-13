import argparse
from utils import read_yaml, classifier_trainer
from models.transformer_models import TransformerEnsemble
from models.mlp_models import EnsembleModel
from dataloaders.dataloaders4mlp import transformer_data4mlp, cnn_data4mlp, mlp_data4mlp
from dataloaders.dataloaders4transformer import transformer_data4transformer, cnn_data4transformer, mlp_data4transformer

parser = argparse.ArgumentParser(description="train an ensemble of mlps on data")
parser.add_argument("-d", "--datatype", type=str, required=True, help="on data (transformer, cnn, mlp)")
parser.add_argument("-m", "--modeltype", type=str, required=True, help="choose model (transformer, mlp)")

# CUDA_VISIBLE_DEVICES=4
def train_mlp4transformer(config):
    train_dataloader, val_dataloader, _, class_counts = transformer_data4mlp(config)
    x,_ = next(iter(train_dataloader))
    features_dim = x.shape[-1]
    classifier_model = EnsembleModel(model_list=[1,2,3,4,5,6,7,1,2,3,4,5,6], input_size=features_dim, output_size=class_counts, max_neurons=8, min_neurons=4, learning_rate=config["learning_rate"])
    _ = classifier_trainer(classifier_model, train_dataloader, val_dataloader, config, filename='mlp4transformer')
    

def train_mlp4cnn(config):
    train_dataloader, val_dataloader, _, class_counts = cnn_data4mlp(config)    
    x,_ = next(iter(train_dataloader))
    features_dim = x.shape[-1]
    classifier_model = EnsembleModel(model_list=[1,2,3,4,5,6,7,8,4,5,6,7,8,4,5,6], input_size=features_dim, output_size=class_counts, max_neurons=8, min_neurons=4, learning_rate=config["learning_rate"])

    _ = classifier_trainer(classifier_model, train_dataloader, val_dataloader, config, filename='mlp4cnn')


def train_mlp4mlp(config):
    train_dataloader, val_dataloader, _, class_counts = mlp_data4mlp(config)
    x,_ = next(iter(train_dataloader))
    features_dim = x.shape[-1]
    classifier_model = EnsembleModel(model_list=[1,2,3,4,5,6,7,1,2,3,4,5,6], input_size=features_dim, output_size=class_counts, max_neurons=8, min_neurons=4, learning_rate=config["learning_rate"])
    _ = classifier_trainer(classifier_model, train_dataloader, val_dataloader, config, filename='mlp4mlp')


def train_transformer4transformer(config):
    transformers_architectures = read_yaml("transformers_architectures.yaml")['model_configs']
    train_dataloader, val_dataloader, _, data_related_info = transformer_data4transformer(config)
    x, z, _ = next(iter(train_dataloader))
    num_features = x.shape[-1]
    extra_features_num = z.shape[-1]
    classifier_model = TransformerEnsemble(model_configs=transformers_architectures, num_features=num_features, 
                                    num_classes=data_related_info["class_counts"], learning_rate=config["learning_rate"], 
                                    max_seq_len=data_related_info["max_seq_len"], 
                                    extra_fetures_num=extra_features_num)   

    _ = classifier_trainer(classifier_model, train_dataloader, val_dataloader, config, filename='transformer4transformer')
    
def train_transformer4cnn(config):
    transformers_architectures = read_yaml("transformers_architectures.yaml")['model_configs']
    train_dataloader, val_dataloader, _, data_related_info = cnn_data4transformer(config)
    x, z, _ = next(iter(train_dataloader))
    num_features = x.shape[-1]
    extra_features_num = z.shape[-1]
    classifier_model = TransformerEnsemble(model_configs=transformers_architectures, num_features=num_features, 
                                    num_classes=data_related_info["class_counts"], learning_rate=config["learning_rate"], 
                                    max_seq_len=data_related_info["max_seq_len"], 
                                    extra_fetures_num=extra_features_num)   

    _ = classifier_trainer(classifier_model, train_dataloader, val_dataloader, config, filename='transformer4cnn')

def train_transformer4mlp(config):
    transformers_architectures = read_yaml("transformers_architectures.yaml")['model_configs']
    train_dataloader, val_dataloader, _, data_related_info = mlp_data4transformer(config)
    x, z, _ = next(iter(train_dataloader))
    num_features = x.shape[-1]
    extra_features_num = z.shape[-1]
    classifier_model = TransformerEnsemble(model_configs=transformers_architectures, num_features=num_features, 
                                    num_classes=data_related_info["class_counts"], learning_rate=config["learning_rate"], 
                                    max_seq_len=data_related_info["max_seq_len"], 
                                    extra_fetures_num=extra_features_num)   

    _ = classifier_trainer(classifier_model, train_dataloader, val_dataloader, config, filename='transformer4mlp')
    
    
if __name__=="__main__":
    config = read_yaml("config.yaml")
    args = parser.parse_args()
    data_to_train_on = args.datatype
    model_type = args.modeltype

    if model_type =="mlp":
        if data_to_train_on=='transformer':
            train_mlp4transformer(config)
        elif data_to_train_on=='cnn':
            train_mlp4cnn(config)
        elif data_to_train_on=='mlp':
            train_mlp4mlp(config)
    elif model_type == "transformer":
        if data_to_train_on=='transformer':
            train_transformer4transformer(config)
        elif data_to_train_on=='cnn':
            train_transformer4cnn(config)
        elif data_to_train_on=='mlp':
            train_transformer4mlp(config)
        