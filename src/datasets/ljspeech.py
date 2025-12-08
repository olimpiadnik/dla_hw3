from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset
import torchaudio

from src.model.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig


class LJSpeechDataset(Dataset):
    def __init__(
        self,
        root: str,
        segment_size: int = 8192,
        sampling_rate: int = 22050,
        return_text: bool = False,
        mel_config: Optional[MelSpectrogramConfig] = None,
    ):
        super().__init__()

        root_path = Path(root)
        candidates = [
            root_path,
            root_path / "LJSpeech-1.1",
        ]

        data_root = None
        for c in candidates:
            if (c / "metadata.csv").exists():
                data_root = c
                break

        if data_root is None:
            raise FileNotFoundError(
                f"metadata.csv not found in any of: {[str(c) for c in candidates]}"
            )

        self.data_root = data_root
        metadata_path = self.data_root / "metadata.csv"
        self.wav_dir = self.data_root / "wavs"

        self.segment_size = segment_size
        self.target_sr = sampling_rate
        self.return_text = return_text

        self.resampler = None
        self._resampler_initialized = False

        # mel-экстрактор
        if mel_config is None:
            mel_config = MelSpectrogramConfig(sr=sampling_rate)
        self.mel_extractor = MelSpectrogram(mel_config)

        self.items: List[Dict[str, Any]] = []
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 2:
                    continue
                utt_id = parts[0]
                text = parts[1]
                wav_path = self.wav_dir / f"{utt_id}.wav"
                if wav_path.exists():
                    self.items.append({"wav_path": wav_path, "text": text})

        if not self.items:
            raise RuntimeError(f"No wav files found under {self.wav_dir}")

    def _maybe_init_resampler(self, orig_sr: int):
        if self._resampler_initialized:
            return
        if orig_sr != self.target_sr:
            self.resampler = torchaudio.transforms.Resample(orig_sr, self.target_sr)
        self._resampler_initialized = True

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        wav_path = item["wav_path"]

        audio, sr = torchaudio.load(wav_path)  # [C, T]
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)

        self._maybe_init_resampler(sr)
        if self.resampler is not None:
            audio = self.resampler(audio)

        audio = audio.squeeze(0)  # [T]
        audio = audio.clamp(-1.0, 1.0)

        if audio.size(-1) >= self.segment_size:
            max_start = audio.size(-1) - self.segment_size
            start = torch.randint(0, max_start + 1, (1,)).item()
            audio = audio[start : start + self.segment_size]
        else:
            pad_size = self.segment_size - audio.size(-1)
            audio = torch.nn.functional.pad(audio, (0, pad_size))

        mel = self.mel_extractor(audio.unsqueeze(0)) 
        mel = mel.squeeze(0)  

        sample: Dict[str, Any] = {
            "audio": audio,   
            "mel": mel,  
            "path": str(wav_path),
        }
        if self.return_text:
            sample["text"] = item["text"]

        return sample
