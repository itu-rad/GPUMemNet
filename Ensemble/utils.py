import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
import os
import shutil
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import json

def save_to_json(data, file_path):
    """
    Saves a Python object to a JSON file, ensuring the directory path exists,
    and converts NumPy arrays to lists for JSON compatibility.
    
    Parameters:
    data (dict or list): The data to be saved.
    file_path (str): The file path, including directory and filename, where the JSON file will be saved.
    """
    # Convert NumPy arrays to lists
    def convert_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_ndarray(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_ndarray(v) for v in obj]
        else:
            return obj

    data = convert_ndarray(data)
    
    # Ensure directory exists
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # Save to JSON
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)



def load_from_json(filename):
    """
    Loads data from a JSON file and returns it as a Python object.
    
    Parameters:
    filename (str): The file path of the JSON file to load.
    
    Returns:
    dict or list: The data loaded from the JSON file.
    """
    with open(filename, 'r') as f:
        return json.load(f)


def read_yaml(file_path):
    """
    Reads a YAML file and returns its contents as a dictionary.

    Parameters:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The contents of the YAML file.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def classifier_trainer(classifier_model, train_dataloader, val_dataloader, config, filename=None):
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=True, mode='min')
    classifier_model.train()
    if filename:
        checkpoint_dir = os.path.join(config['save_model_dir'], filename)
        
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        
        checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir, 
                monitor="val_loss",           # Metric to monitor
                mode="min",                   # 'min' for loss, 'max' for accuracy/score
                filename=filename,   # Custom filename for best model
                save_top_k=1,                 # Save only the best model
                verbose=True                  # Print saving info
            )
        callbacks = [early_stopping, checkpoint_callback]
    else:
        callbacks = [early_stopping]
    trainer = pl.Trainer(
        max_epochs=600,
        callbacks=callbacks,
    )
    # Train the model
    trainer.fit(classifier_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    return classifier_model

def below_diagonal_sum_rate(confusion_matrix):
    confusion_matrix = np.array(confusion_matrix)
    below_diag_sum = np.sum(np.tril(confusion_matrix, k=-1))
    total_data_points = np.sum(confusion_matrix)
    rate = below_diag_sum / total_data_points
    return rate

def calculate_metrics(gt, preds):
    # Calculate the various metrics
    accuracy = accuracy_score(gt, preds)
    precision = precision_score(gt, preds, average='weighted')  # Use 'weighted' to handle class imbalance
    recall = recall_score(gt, preds, average='weighted')
    f1 = f1_score(gt, preds, average='weighted')
    cm = confusion_matrix(gt, preds)

    # Print the results
    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-Score: {:.4f}".format(f1))
    print("Below Diagonal Sum Rate: {:.4f}".format(below_diagonal_sum_rate(cm)))
    
    print("\nConfusion Matrix:")
    print(cm)

    # Full classification report
    print("\nClassification Report:")
    classification_r = classification_report(gt, preds)
    print(classification_r)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1":f1, 
            "below_diagonal_percentage":below_diagonal_sum_rate(cm), "confusion_matrix":cm, "classification_report":classification_r}


def model_eval(classifier_model, test_dataloader, transformer=False):
    classifier_model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier_model.to(device)
    pred_list, gt = list(), list()
    with torch.no_grad():
        for test_data in test_dataloader:
            if transformer:
                pred = classifier_model(test_data[0].to(device), test_data[1].to(device))
                gt.append(test_data[2].item())
            else:
                pred = classifier_model(test_data[0].to(device))
                gt.append(test_data[1].item())
            pred_list.append(torch.argmax(pred, 1).item())

    classification_metrics = calculate_metrics(gt, pred_list)
    return classification_metrics