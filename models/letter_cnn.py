import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import AttentionPooling

class LetterCNN(nn.Module):
    """
    CNN + Attention pooling for isolated letter recognition (A–Z).
    Returns letter prediction + confidence score.
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

        # Compute feature size dynamically
        self.feature_size = self._get_feature_size(n_mels)

        # Attention pooling
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

    def _get_feature_size(self, n_mels):
        dummy = torch.randn(1, 1, n_mels, 100)
        x = self._forward_features(dummy)
        b, c, f, t = x.shape
        return c * f

    def forward(self, x):
        x = self._forward_features(x)
        x = x.flatten(1, 2)        # merge (channels, mel_bins//8) into features
        x = self.attention(x)      # attention pooling over time
        x = self.classifier(x)
        return x

    def predict_letter(self, x):
        """Predict a single letter without confidence"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            preds = torch.argmax(logits, dim=1)
            letters = [chr(ord('A') + p.item()) for p in preds]
            return letters[0] if len(letters) == 1 else letters

    def predict_with_confidence(self, x):
        """
        Predict letter(s) with confidence score(s).
        
        Returns:
            - single input: (letter, confidence)
            - batch: (list of letters, list of confidences)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, preds = torch.max(probs, dim=1)

            letters = [chr(ord('A') + p.item()) for p in preds]
            scores = confidence.tolist()

            if len(letters) == 1:
                return letters[0], scores[0]
            return letters, scores
