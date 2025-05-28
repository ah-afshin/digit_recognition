import ssl
import os

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader



def get_dataloaders(batch_size: int=32) -> tuple[DataLoader, DataLoader]:
    """Downloads the MNIST dataset (if not already present) and returns DataLoader objects.

    This function applies a basic transformation (`ToTensor`) to convert the images 
    into PyTorch tensors and prepares `DataLoader` instances for both training and test sets.
    
    Describtion of Current Script: Robust MNIST dataloader that uses reliable Google Cloud URLs.
    
    Args:
        batch_size (int, optional): Number of samples per batch. Defaults to 32.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing train and test DataLoaders.
    """
    transform = transforms.ToTensor()
    
    # Bypass SSL verification (temporarily for download only)
    original_ssl_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    
    try:
        # Override the URLs completely
        datasets.MNIST.urls = [
            'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
            'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz',
            'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
            'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz',
        ]
        
        # Create data directory if it doesn't exist
        os.makedirs("data/MNIST/raw", exist_ok=True)
        
        train_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform
        )
        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transform
        )
        
    finally:
        # Restore original SSL context
        ssl._create_default_https_context = original_ssl_context
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


if __name__ == "__main__":
    """download data if does not exists"""
    _ = get_dataloaders()
