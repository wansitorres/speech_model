import torch
import torchaudio

class AudioProcessor:
    """Convert audio → log-mel spectrogram"""

    def __init__(self, sample_rate=16000, n_mels=64):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels,
            fmin=0, fmax=8000
        )

    def process_audio(self, audio_path):
        waveform, orig_sr = torchaudio.load(audio_path)
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Extract mel spectrogram
        mel = self.mel_transform(waveform)
        log_mel = torch.log(mel + 1e-8)

        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

        return log_mel.unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
