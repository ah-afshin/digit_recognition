from sys import argv
import torch as t

from models import DigitRecognizerMLP
from dataloaders import get_dataloaders
from train import train_model
from test import test_model
from ui import run_ui


# Hyperparameters
BATCH_SIZE = 32
HIDDEN_UINTS = 128
EPOCHS = 5
LEARNING_RATE = 0.005


def main(model_name: str|None) -> None:
    """Main training pipeline for digit recognition model.

    Args:
        model_name (str or None): Optional name for saving the trained model.
                                  If None, a name is generated automatically based on hyperparameters.

    Returns:
        None
    """
    device = "cuda" if t.cuda.is_available() else "cpu"
    train_dl, test_dl = get_dataloaders(BATCH_SIZE)

    model = DigitRecognizerMLP(HIDDEN_UINTS).to(device)
    train_model(model, train_dl, epochs=EPOCHS, device=device, learning_rate=LEARNING_RATE)
    test_model(model, test_dl, device=device)

    if model_name is None:
        model_name = f"mnist_recog_model__B{BATCH_SIZE}_hu{HIDDEN_UINTS}_epoch{EPOCHS}_lr{LEARNING_RATE}"
    t.save(model.state_dict(), model_name+".pth")



if __name__ == "__main__":
    if (len(argv) > 1) and (argv[1] == "ui"):
        # show GUI for users test
        run_ui()
    
    else:
        # train model
        model_name = argv[1] if len(argv)>1 else None
        main(model_name)
