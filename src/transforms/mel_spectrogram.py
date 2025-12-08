from typing import Dict, Any, Optional

import torch

from src.model.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig


class AddMelSpectrogram:
    def __init__(
        self,
        sr: int = 22050,
        win_length: int = 1024,
        hop_length: int = 256,
        n_fft: int = 1024,
        f_min: int = 0,
        f_max: int = 8000,
        n_mels: int = 80,
        power: float = 1.0,
        audio_key: str = "audio",
        mel_key: str = "mel",
    ):
        self.audio_key = audio_key
        self.mel_key = mel_key

        config = MelSpectrogramConfig(
            sr=sr,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=power,
        )
        self.mel_extractor = MelSpectrogram(config)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        audio = sample[self.audio_key]  # Tensor

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  
        elif audio.dim() == 2 and audio.size(0) == 1:
            pass
        elif audio.dim() == 2 and audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        elif audio.dim() == 3:
            if audio.size(1) > 1:
                audio = audio.mean(dim=1)
        else:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")

        mel = self.mel_extractor(audio) 
        if mel.size(0) == 1:
            mel_out = mel.squeeze(0)
        else:
            mel_out = mel

        sample[self.mel_key] = mel_out
        return sample
