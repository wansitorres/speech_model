import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from models.letter_cnn import LetterCNN
from data import LetterDataset
from utils import AudioProcessor, load_model


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total * 100
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss


def evaluate_with_confidence(model, test_loader, device):
    """Evaluate model with confidence analysis"""
    model.eval()
    
    correct = 0
    total = 0
    confidence_scores = []
    correct_confidences = []
    incorrect_confidences = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Get predictions with confidence
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            correct_mask = (predictions == y)
            correct += correct_mask.sum().item()
            total += y.size(0)
            
            for i in range(len(predictions)):
                conf = confidences[i].item()
                confidence_scores.append(conf)
                
                if correct_mask[i].item():
                    correct_confidences.append(conf)
                else:
                    incorrect_confidences.append(conf)
    
    accuracy = correct / total * 100
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    avg_correct_conf = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0
    avg_incorrect_conf = sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0
    
    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'avg_correct_confidence': avg_correct_conf,
        'avg_incorrect_confidence': avg_incorrect_conf,
        'confidence_separation': avg_correct_conf - avg_incorrect_conf
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model("best_letter_model.pth", device=device)
    
    # Load test dataset
    processor = AudioProcessor(n_mels=64)
    test_dataset = LetterDataset("dataset_root", processor, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Basic evaluation
    accuracy, loss = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.1f}%")
    print(f"Test Loss: {loss:.3f}")
    
    # Confidence evaluation
    conf_metrics = evaluate_with_confidence(model, test_loader, device)
    print(f"\nConfidence Analysis:")
    print(f"Average Confidence: {conf_metrics['avg_confidence']:.3f}")
    print(f"Correct Predictions Confidence: {conf_metrics['avg_correct_confidence']:.3f}")
    print(f"Incorrect Predictions Confidence: {conf_metrics['avg_incorrect_confidence']:.3f}")
    print(f"Confidence Separation: {conf_metrics['confidence_separation']:.3f}")