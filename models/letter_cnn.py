import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import AttentionPooling

class LetterCNN(nn.Module):
    """
    CNN + Attention pooling for isolated letter recognition (A–Z).
    """

    def __init__(self, n_mels=64, n_classes=26):
        super().__init__()

        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)

        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)

        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)

        # Attention pooling
        reduced_mels = n_mels // 8  # after 3x pooling
        self.feature_size = 128 * reduced_mels
        self.attention = AttentionPooling(in_features=self.feature_size)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes)
        )

    def _forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(self.pool1(x))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(self.pool2(x))

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(self.pool3(x))
        return x  # (batch, 128, mel_bins//8, time)

    def forward(self, x):
        x = self._forward_features(x)
        x = x.flatten(1, 2)        # merge (channels, mel_bins) into features
        x = self.attention(x)      # attention pooling over time
        x = self.classifier(x)
        return x

    def predict_letter(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            pred = torch.argmax(logits, dim=1)
            return chr(ord('A') + pred.item())
