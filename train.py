import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from models.letter_cnn import LetterCNN
from data import AudioProcessor, LetterDataset
from models import LetterCNN


def train_model(model, train_loader, val_loader, epochs=30, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0
    for epoch in range(epochs):
        # Training
        model.train()
        total, correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        train_acc = correct / total * 100

        # Validation
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total * 100

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_letter_model.pth")

        print(f"Epoch {epoch+1}: Train {train_acc:.1f}%, Val {val_acc:.1f}%")

    print(f"Best validation accuracy: {best_acc:.1f}%")

if __name__ == "__main__":
    processor = AudioProcessor(n_mels=64)
    dataset = LetterDataset("dataset_root", processor)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    # Model
    model = LetterCNN(n_mels=64, n_classes=26)

    # Train
    train_model(model, train_loader, val_loader, epochs=30)
