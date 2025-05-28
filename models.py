import torch as t
from torch import nn



class DigitRecognizerMLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) model for handwritten digit recognition.

    This model is designed for classification of digits from the MNIST dataset.
    It consists of multiple fully connected layers with ReLU activation.

    Attributes:
        mlp_model (nn.Sequential): The sequential MLP architecture.
    
    Args:
        hidden_units (int): Number of neurons in each hidden layer.
    """

    def __init__(self, hidden_units: int):
        """Initializes the MLP model architecture.

        Args:
            hidden_units (int): Number of neurons in each hidden layer.
        """
        super().__init__()
        self.mlp_model = nn.Sequential(
            nn.Flatten(), # input from mnist dataset is a tesor.
            
            nn.Linear(28*28, hidden_units),
            nn.ReLU(),
            
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            
            nn.Linear(hidden_units, 10)
        )
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        """Defines (overrides) the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor containing an image.

        Returns:
            torch.Tensor: The logits corresponding to digit classification.
        """
        return self.mlp_model(x)
