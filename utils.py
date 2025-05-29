import re

import torch as t

from models import DigitRecognizerMLP



def accuracy_func(y_true, y_pred):
    """Calculate accuracy between predicted and true labels.

    Args:
        y_true (Tensor): True labels, shape (batch_size,)
        y_pred (Tensor): Predicted labels, shape (batch_size,)

    Returns:
        float: Accuracy percentage (0-100)
    """
    correct = t.eq(y_true, y_pred).sum().item()
    total = y_true.size(0)
    accuracy = (correct/total) * 100 # get the percentage
    return accuracy


def load_model(model_name: str, device: str) -> DigitRecognizerMLP:
    hu = re.search(r'hu(\d+)', model_name)
    if hu:
        hidden_units = int(hu.group(1))
    else:
        raise ValueError(f"Could not extract hidden_units from model name: {model_name}")
    
    loaded_model = DigitRecognizerMLP(hidden_units).to(device)
    loaded_model.load_state_dict(t.load(f=model_name))
    return loaded_model
