import torch as t
from torch import nn
from torch.utils.data import DataLoader

from models import DigitRecognizerMLP



def train_model(
        model: DigitRecognizerMLP,
        train_dl: DataLoader,
        epochs: int,
        device: str,
        learning_rate: float
    ) -> None:
    """Train the given DigitRecognizerMLP model using provided training data.

    Args:
        model (DigitRecognizerMLP): The neural network model to train.
        train_dl (DataLoader): PyTorch DataLoader containing training data.
        epochs (int): Number of times to loop over the full dataset.
        device (str): Device to use for training ('cpu' or 'cuda').
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        None
    """
    loss_func = nn.CrossEntropyLoss()
    optim = t.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss_of_epoch = 0

        for batch in train_dl:
            x, y = batch
            x = x.view(x.size(0), -1) # Flatten image from [B, 1, 28, 28] to [B, 784]
            x, y = x.to(device), y.to(device)

            y_pred_logit = model(x)
            loss = loss_func(y_pred_logit, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss_of_epoch += loss.item()
        print(f"Epoch {epoch+1} | loss: {total_loss_of_epoch}")
