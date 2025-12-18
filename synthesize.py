import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import soundfile as sf 
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from hydra.utils import instantiate

import torchaudio

from src.model import HiFiGAN 
from src.model import MelSpectrogram, MelSpectrogramConfig

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def save_wav(path: Path, audio, sr: int):

    if not isinstance(audio, torch.Tensor):
        audio_t = torch.as_tensor(audio, dtype=torch.float32)
    else:
        audio_t = audio.detach().cpu()

    while audio_t.dim() > 1 and audio_t.shape[0] == 1 and audio_t.dim() > 2:
        audio_t = audio_t.squeeze(0)

    if audio_t.dim() == 1:
        # [T] -> [1, T]
        audio_t = audio_t.unsqueeze(0)
    elif audio_t.dim() == 2:
        pass
    else:
        audio_t = audio_t.reshape(1, -1)

    torchaudio.save(str(path), audio_t, sample_rate=sr)


def load_config_and_model(
    checkpoint_path: str,
    config_path: Optional[str],
    device: torch.device,
) -> Tuple[torch.nn.Module, MelSpectrogram]:
    ckpt_path = Path(checkpoint_path)
    if config_path is None:
        config_path = ckpt_path.parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    if "model" not in cfg:
        raise KeyError("Config has no 'model' section; cannot instantiate HiFi-GAN")

    model_cfg = cfg.model
    model: torch.nn.Module = instantiate(model_cfg)

    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    if hasattr(model, "mel_transform"):
        mel_extractor = model.mel_transform
        mel_extractor.to(device)
        mel_extractor.eval()
    else:
        mel_cfg = MelSpectrogramConfig()
        mel_extractor = MelSpectrogram(mel_cfg).to(device).eval()

    return model, mel_extractor


def collect_audio_files(root: str, exts=(".wav", ".flac", ".ogg")) -> List[Path]:
    root_path = Path(root)
    files: List[Path] = []
    for p in root_path.rglob("*"):
        if p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def resynthesize_directory(
    model: torch.nn.Module,
    mel_extractor: MelSpectrogram,
    input_audio_dir: str,
    output_dir: str,
    device: torch.device,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = collect_audio_files(input_audio_dir)
    if not audio_files:
        raise RuntimeError(f"No audio files found in {input_audio_dir}")

    sr = 22050
    if hasattr(mel_extractor, "config") and hasattr(mel_extractor.config, "sr"):
        sr = mel_extractor.config.sr

    print(f"Found {len(audio_files)} files. Writing resynth audio to {out_dir}, sr={sr}")

    mel_extractor = mel_extractor.to(device).eval()

    for wav_path in audio_files:
        wav, wav_sr = torchaudio.load(str(wav_path))  # [C, T]

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if wav_sr != sr:
            wav = torchaudio.functional.resample(wav, wav_sr, sr)

        wav = wav.to(device)
        wav = wav.squeeze(0).unsqueeze(0)  # [1, T]

        with torch.no_grad():
            mel = mel_extractor(wav)  # [1, n_mels, Tm]
            if hasattr(model, "generate"):
                audio_hat = model.generate(mel)
            elif hasattr(model, "generator"):
                audio_hat = model.generator(mel)
            else:
                out = model(audio=None, mel=mel)
                audio_hat = out["audio_hat"]

        audio = audio_hat.squeeze(0).detach().cpu()
        out_path = out_dir / f"{wav_path.stem}.wav"
        save_wav(out_path, audio, sr)
        print(f"[resynthesize] {wav_path.name} -> {out_path.name}")

def load_espnet_tts(model_name: str, device: torch.device):
    try:
        from espnet2.bin.tts_inference import Text2Speech
    except Exception as e:
        raise ImportError(
            "ESPNet is not installed. Install with `pip install espnet espnet_model_zoo`."
        ) from e

    dev_str = "cuda" if device.type == "cuda" else "cpu"
    print(f"Loading ESPNet TTS model: {model_name}")
    tts = Text2Speech.from_pretrained(
        model_name,
        device=dev_str,
    )
    return tts


def texts_to_espnet_mels(
    texts: List[str],
    tts,
    mel_extractor: MelSpectrogram,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mels: List[torch.Tensor] = []
    lengths: List[int] = []

    mel_extractor = mel_extractor.to(device).eval()

    with torch.no_grad():
        for text in texts:
            out_dict = tts(text)

            feat = None
            if "feat_gen_denorm" in out_dict:
                feat = out_dict["feat_gen_denorm"]
            elif "feat_gen" in out_dict:
                feat = out_dict["feat_gen"]

            if feat is not None:
                if not isinstance(feat, torch.Tensor):
                    feat = torch.as_tensor(feat, dtype=torch.float32, device=device)
                else:
                    feat = feat.to(device)

                if feat.dim() == 3:
                    feat = feat.squeeze(0)
                if feat.dim() != 2:
                    raise RuntimeError(f"Unexpected feat shape from ESPNet: {feat.shape}")

                mel = feat.transpose(0, 1)  # [T, C] -> [C, T]

            else:
                if "wav" not in out_dict:
                    raise RuntimeError(
                        "ESPNet Text2Speech output has no 'feat_gen', 'feat_gen_denorm' or 'wav'."
                    )
                wav = out_dict["wav"]
                if not isinstance(wav, torch.Tensor):
                    wav = torch.as_tensor(wav, dtype=torch.float32)
                if wav.dim() > 1:
                    wav = wav.squeeze(0)

                wav = wav.to(device).unsqueeze(0) 
                mel_full = mel_extractor(wav)  
                mel = mel_full[0]  

            mels.append(mel)
            lengths.append(mel.shape[-1])

    max_len = max(lengths)
    C = mels[0].shape[0]
    B = len(mels)

    padded = []
    for mel, L in zip(mels, lengths):
        if L < max_len:
            pad = max_len - L
            pad_value = float(mel.min().item())
            mel_padded = F.pad(mel, (0, pad), value=pad_value)
        else:
            mel_padded = mel
        padded.append(mel_padded)

    mel_batch = torch.stack(padded, dim=0)  
    mel_lengths = torch.tensor(lengths, device=device, dtype=torch.long)
    return mel_batch, mel_lengths

def load_text_dataset(custom_dir: str) -> List[Tuple[str, str]]:
    root = Path(custom_dir)
    trans_dir = root / "transcriptions"
    if not trans_dir.exists():
        raise FileNotFoundError(f"transcriptions/ not found in {root}")

    items: List[Tuple[str, str]] = []
    for txt_path in sorted(trans_dir.glob("*.txt")):
        utt_id = txt_path.stem
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        items.append((utt_id, text))

    if not items:
        raise RuntimeError(f"No .txt files with text found in {trans_dir}")

    return items


def tts_dataset(
    model: torch.nn.Module,
    tts,
    mel_extractor: MelSpectrogram,
    custom_dir: str,
    output_dir: str,
    device: torch.device,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_text_dataset(custom_dir)
    print(f"Loaded {len(items)} texts from {custom_dir}")

    sr = 22050
    if hasattr(model, "mel_transform") and hasattr(model.mel_transform, "config"):
        sr = getattr(model.mel_transform.config, "sr", sr)

    for utt_id, text in items:
        print(f"[tts_dataset] {utt_id}: '{text}'")

        mels, _ = texts_to_espnet_mels([text], tts, mel_extractor, device)  # [1, C, Tm]
        with torch.no_grad():
            if hasattr(model, "generate"):
                audio_hat = model.generate(mels)
            elif hasattr(model, "generator"):
                audio_hat = model.generator(mels)
            else:
                out = model(audio=None, mel=mels)
                audio_hat = out["audio_hat"]

        audio = audio_hat.squeeze(0).detach().cpu()
        out_path = out_dir / f"{utt_id}.wav"
        save_wav(out_path, audio, sr)
        print(f"[tts_dataset] saved: {out_path}")


def tts_single(
    model: torch.nn.Module,
    tts,
    mel_extractor: MelSpectrogram,
    text: str,
    output_path: str,
    device: torch.device,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sr = 22050
    if hasattr(model, "mel_transform") and hasattr(model.mel_transform, "config"):
        sr = getattr(model.mel_transform.config, "sr", sr)

    print(f"[tts_single] text: '{text}'")
    mels, _ = texts_to_espnet_mels([text], tts, mel_extractor, device)  # [1, C, Tm]
    with torch.no_grad():
        if hasattr(model, "generate"):
            audio_hat = model.generate(mels)
        elif hasattr(model, "generator"):
            audio_hat = model.generator(mels)
        else:
            out = model(audio=None, mel=mels)
            audio_hat = out["audio_hat"]

    audio = audio_hat.squeeze(0).detach().cpu()
    save_wav(output_path, audio, sr)
    print(f"[tts_single] saved: {output_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HiFi-GAN vocoder synthesis script (resynthesize / TTS via ESPNet)."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to HiFi-GAN checkpoint (*.pth).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to config.yaml for the checkpoint "
        "(default: <checkpoint_dir>/config.yaml).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["resynthesize", "tts_dataset", "tts_single"],
        help="Inference mode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='"auto", "cpu" или "cuda".',
    )

    parser.add_argument(
        "--input-audio-dir",
        type=str,
        help="Directory with input audio files for resynthesis.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for saving synthesized audio.",
    )

    parser.add_argument(
        "--custom-dir",
        type=str,
        help="Custom dataset dir with 'transcriptions/*.txt' for tts_dataset mode.",
    )

    parser.add_argument(
        "--text",
        type=str,
        help="Single text for TTS (tts_single mode).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output wav path for tts_single mode.",
    )

    parser.add_argument(
        "--espnet_model",
        type=str,
        default="espnet/kan-bayashi_ljspeech_vits",
        help="ESPNet model name for Text2Speech.from_pretrained (used in TTS modes).",
    )

    args = parser.parse_args()

    if args.mode == "resynthesize":
        if args.input_audio_dir is None or args.output_dir is None:
            parser.error(
                "--input-audio-dir и --output-dir обязательны для mode=resynthesize"
            )
    elif args.mode == "tts_dataset":
        if args.custom_dir is None or args.output_dir is None:
            parser.error("--custom-dir и --output-dir обязательны для mode=tts_dataset")
    elif args.mode == "tts_single":
        if args.text is None or args.output_path is None:
            parser.error("--text и --output-path обязательны для mode=tts_single")

    return args


def main():
    args = parse_args()
    device = resolve_device(args.device)

    print(f"Using device: {device}")
    print(f"Loading model from checkpoint: {args.checkpoint_path}")

    model, mel_extractor = load_config_and_model(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        device=device,
    )
    if args.mode == "resynthesize":
        resynthesize_directory(
            model=model,
            mel_extractor=mel_extractor,
            input_audio_dir=args.input_audio_dir,
            output_dir=args.output_dir,
            device=device,
        )
        return

    tts = load_espnet_tts(args.espnet_model, device=device)

    if args.mode == "tts_dataset":
        tts_dataset(
            model=model,
            tts=tts,
            mel_extractor=mel_extractor,
            custom_dir=args.custom_dir,
            output_dir=args.output_dir,
            device=device,
        )
    elif args.mode == "tts_single":
        tts_single(
            model=model,
            tts=tts,
            mel_extractor=mel_extractor,
            text=args.text,
            output_path=args.output_path,
            device=device,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
