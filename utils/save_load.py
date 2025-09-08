import torch
import os
from models.letter_cnn import LetterCNN

def save_model(model, path="checkpoints/best_letter_model.pth"):
    """
    Save trained LetterCNN model to a file.
    
    Args:
        model (torch.nn.Module): Trained model
        path (str): Save path (.pth)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to {path}")


def load_model(path="checkpoints/best_letter_model.pth", n_mels=64, n_classes=26, device="cpu"):
    """
    Load LetterCNN model from a checkpoint.
    
    Args:
        path (str): Path to checkpoint (.pth)
        n_mels (int): Mel bins (must match training)
        n_classes (int): Number of output classes (default=26)
        device (str): 'cpu' or 'cuda'
    
    Returns:
        model (LetterCNN): Loaded model ready for inference
    """
    model = LetterCNN(n_mels=n_mels, n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"[INFO] Model loaded from {path} (device={device})")
    return model
