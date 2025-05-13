from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from models.transformer_models import TransformerEnsemble
from models.mlp_models import EnsembleModel
from dataloaders.dataloaders4mlp import transformer_data4mlp, cnn_data4mlp, mlp_data4mlp
from dataloaders.dataloaders4transformer import transformer_data4transformer, cnn_data4transformer, mlp_data4transformer
from utils import read_yaml, classifier_trainer, model_eval, save_to_json
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(description="k-fold ensemble of mlps on data")
parser.add_argument("-d", "--datatype", type=str, required=True, help="k-fold validation on data (transformer, cnn, mlp)")
parser.add_argument("-m", "--modeltype", type=str, required=True, help="choose model (transformer, mlp)")

def kfold_results(classification_metrics_list):
    print("#"*10 + " K-fold results " + "#"*10)
    metric_names = list(classification_metrics_list[0].keys())
    for metric_name in metric_names:
        if metric_name not in ["confusion_matrix", "classification_report"]:
            kfold_metric = [x[metric_name] for x in classification_metrics_list]
            print(f"{metric_name}: mean: {np.mean(np.array(kfold_metric))}, std: {np.std(np.array(kfold_metric))}")


def perform_kfold(classifier_model, whole_dataset, save_dir, n_splits = 3, for_transformer=False):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_indices = np.arange(len(whole_dataset))
    fold_splits = list(kf.split(all_indices))
    classification_metrics_list = list()

    for fold in range(n_splits):
        print(f"Fold {fold + 1}/{n_splits}")
        fold_indices = fold_splits[fold][1]
        n_test = int(len(fold_indices) * 0.70)
        n_val = int(len(fold_indices) * 0.30)
        
        test_indices = fold_indices[:n_test]
        val_indices = fold_indices[n_test:]
        train_indices = np.setdiff1d(all_indices, np.union1d(test_indices, val_indices))

        train_subset = Subset(whole_dataset, train_indices)
        val_subset = Subset(whole_dataset, val_indices)
        test_subset = Subset(whole_dataset, test_indices)

        train_dataloader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=config["batch_size"])
        test_dataloader = DataLoader(test_subset, batch_size=1)

        
        classifier_model = classifier_trainer(classifier_model, train_dataloader, val_dataloader, config)
        metrics = model_eval(classifier_model, test_dataloader, transformer=for_transformer)
        classification_metrics_list.append(metrics)

    save_to_json(classification_metrics_list, save_dir)
    kfold_results(classification_metrics_list)


if __name__ == "__main__":
    config = read_yaml("config.yaml")
    args = parser.parse_args()
    data_to_test_on = args.datatype
    model_type = args.modeltype

    save_dir = os.path.join("./k_fold_results",f"{model_type}_model" ,data_to_test_on +'.json')
    
    if model_type =="mlp":
        for_transformer = False
        if data_to_test_on=='transformer':
            whole_dataset, class_counts = transformer_data4mlp(config, kfold=True)
            x,_ = next(iter(whole_dataset))
            features_dim = x.shape[-1]
            classifier_model = EnsembleModel(model_list=[1,2,3,4,5,6,7,1,2,3,4,5,6], input_size=features_dim, output_size=class_counts, max_neurons=8, min_neurons=4, learning_rate=config["learning_rate"])
        elif data_to_test_on=='cnn':
            whole_dataset, class_counts = cnn_data4mlp(config, kfold=True)
            x,_ = next(iter(whole_dataset))
            features_dim = x.shape[-1]
            classifier_model = EnsembleModel(model_list=[1,2,3,4,5,6,7,8,4,5,6,7,8,4,5,6], input_size=features_dim, output_size=class_counts, max_neurons=8, min_neurons=4, learning_rate=config["learning_rate"])
        elif data_to_test_on=='mlp':
            whole_dataset, class_counts = mlp_data4mlp(config, kfold=True)
            x,_ = next(iter(whole_dataset))
            features_dim = x.shape[-1]
            classifier_model = EnsembleModel(model_list=[1,2,3,4,5,6,7,1,2,3,4,5,6], input_size=features_dim, output_size=class_counts, max_neurons=8, min_neurons=4, learning_rate=config["learning_rate"])
        else:
            exit()
    elif model_type == "transformer":
        for_transformer = True
        transformers_architectures = read_yaml("transformers_architectures.yaml")['model_configs']
        if data_to_test_on=='transformer':
            whole_dataset, data_related_info = transformer_data4transformer(config, kfold=True)
            x, z, _ = next(iter(whole_dataset))
            num_features = x.shape[-1]
            extra_features_num = z.shape[-1]
            classifier_model = TransformerEnsemble(model_configs=transformers_architectures, num_features=num_features, 
                                            num_classes=data_related_info["class_counts"], learning_rate=config["learning_rate"], 
                                            max_seq_len=data_related_info["max_seq_len"], 
                                            extra_fetures_num=extra_features_num)  
        elif data_to_test_on=='cnn':
            whole_dataset, data_related_info = cnn_data4transformer(config, kfold=True)
            x, z, _ = next(iter(whole_dataset))
            num_features = x.shape[-1]
            extra_features_num = z.shape[-1]
            classifier_model = TransformerEnsemble(model_configs=transformers_architectures, num_features=num_features, 
                                            num_classes=data_related_info["class_counts"], learning_rate=config["learning_rate"], 
                                            max_seq_len=data_related_info["max_seq_len"], 
                                            extra_fetures_num=extra_features_num)  
        elif data_to_test_on=='mlp':
            whole_dataset, data_related_info = mlp_data4transformer(config, kfold=True)
            x, z, _ = next(iter(whole_dataset))
            num_features = x.shape[-1]
            extra_features_num = z.shape[-1]
            classifier_model = TransformerEnsemble(model_configs=transformers_architectures, num_features=num_features, 
                                            num_classes=data_related_info["class_counts"], learning_rate=config["learning_rate"], 
                                            max_seq_len=data_related_info["max_seq_len"], 
                                            extra_fetures_num=extra_features_num)   

    perform_kfold(classifier_model, whole_dataset, save_dir, for_transformer=for_transformer)