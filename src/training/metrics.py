import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report


def compute_metrics(predictions, targets):
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    acc = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='binary', zero_division=0)
    precision = precision_score(targets, predictions, average='binary', zero_division=0)
    recall = recall_score(targets, predictions, average='binary', zero_division=0)
    cm = confusion_matrix(targets, predictions)

    return {
        'accuracy': acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm
    }


def get_classification_report(predictions, targets):
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    return classification_report(targets, predictions)
