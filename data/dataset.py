import os
import torch
from torch.utils.data import Dataset
import torchaudio
import random

class LetterDataset(Dataset):
    """
    Dataset for isolated letter audio.
    Expects: root/A/*.wav, root/B/*.wav, ..., root/Z/*.wav
    """

    def __init__(self, root_dir, processor, augment=False, extensions=(".wav", ".mp3")):
        self.processor = processor
        self.augment = augment
        self.samples = []
        self.label_map = {chr(ord('A')+i): i for i in range(26)}

        for letter, idx in self.label_map.items():
            letter_dir = os.path.join(root_dir, letter)
            if not os.path.isdir(letter_dir):
                continue
            for fname in os.listdir(letter_dir):
                if fname.lower().endswith(extensions):
                    path = os.path.join(letter_dir, fname)
                    self.samples.append((path, idx))

        # Augmentation transforms
        self.noise_factor = 0.005
        self.pitch_shift = torchaudio.transforms.PitchShift(16000, n_steps=2)
        self.speed_perturb = [
            torchaudio.transforms.Resample(16000, int(16000 * 0.9)),
            torchaudio.transforms.Resample(16000, int(16000 * 1.1))
        ]

    def __len__(self):
        return len(self.samples)

    def add_noise(self, waveform):
        noise = torch.randn_like(waveform) * self.noise_factor
        return waveform + noise

    def augment_waveform(self, waveform):
        if random.random() < 0.3:
            waveform = self.add_noise(waveform)
        if random.random() < 0.3:
            waveform = self.pitch_shift(waveform)
        if random.random() < 0.3:
            waveform = random.choice(self.speed_perturb)(waveform)
        return waveform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform, sr = torchaudio.load(path)

        if self.augment:
            waveform = self.augment_waveform(waveform)

        spec = self.processor.mel_transform(waveform)  # mel-spectrogram
        log_mel = torch.log(spec + 1e-8)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        log_mel = log_mel.unsqueeze(0)  # (1, n_mels, time)

        return log_mel, torch.tensor(label, dtype=torch.long)
