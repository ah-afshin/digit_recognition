import torch as t


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
