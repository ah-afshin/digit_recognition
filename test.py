import torch as t
from torch import nn
from torch.utils.data import DataLoader

from models import DigitRecognizerMLP
from utils import accuracy_func


def test_model(
        model: DigitRecognizerMLP,
        test_dl: DataLoader,
        device: str) -> None:
    """Evaluate the trained model on test dataset.

    Args:
        model (DigitRecognizerMLP): The trained neural network model.
        test_dl (DataLoader): DataLoader for test dataset.
        device (str): Device to use for evaluation ('cpu' or 'cuda').

    Returns:
        None: Prints average loss and accuracy.
    """
    loss_func = nn.CrossEntropyLoss()
    
    batch_num = 0
    accuracy_sum = 0
    loss_sum =0

    model.eval()
    with t.inference_mode():

        for batch in test_dl:
            x, y = batch
            x = x.view(x.size(0), -1) # Flatten image from [B, 1, 28, 28] to [B, 784]
            x, y = x.to(device), y.to(device)

            y_pred_logit = model(x)
            y_pred_label = t.softmax(y_pred_logit, dim=1).argmax(dim=1) # we want a list for labels

            batch_num += 1
            loss_sum += loss_func(y_pred_logit, y).item()
            accuracy_sum += accuracy_func(y, y_pred_label)
    
    av_loss = loss_sum/batch_num
    av_acc = accuracy_sum/batch_num
    print("\n\tloss: ", av_loss)
    print("\taccuracy: ", av_acc)
