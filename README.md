# HiFi-GAN Vocoder (DLA HW3, Deep Learning in Audio @ HSE)

This repository contains my **from-scratch implementation of the HiFi-GAN vocoder** for Homework 3 of the *Deep Learning in Audio* course at HSE.

The project is based on the official  
[`pytorch_project_template`](https://github.com/Blinorot/pytorch_project_template)  
and follows its structure: Hydra configs, `train.py` entrypoint, modular `src/` layout, logger abstraction, etc.

The vocoder is trained on **LJSpeech 1.1** and used as the final waveform synthesizer in a TTS pipeline:

> **text → mel-spectrogram (external acoustic model, ESPNet) → waveform (this HiFi-GAN)**


## Installation

> **Recommended**: use a clean environment (conda/venv/Colab). Kaggle notebooks also work (training was done there).

```bash
git clone https://github.com/olimpiadnik/dla_hw3.git
cd dla_hw3

pip install -r requirements.txt
python -m nltk.downloader averaged_perceptron_tagger_eng
```
The requirements.txt contains all the packages needed for:
	•	training HiFi-GAN,
	•	logging with Comet ML,
	•	computing Mel-spectrograms (torchaudio + librosa),
	•	downloading resources from Yandex Disk (yadisk).

Extra dependencies for full TTS (ESPNet)

For the full TTS pipeline (text → mel using an acoustic model → this vocoder), I use an ESPNet Text2Speech model.

These dependencies are not required for training or for resynthesis mode.
Install them only if you want to run TTS modes in synthesize.py:
```
pip install "espnet==202310" "g2p_en" "phonemizer" "torch_complex"
```
All necessary model checkpoints and logs are stored externally (Yandex Disk) and can be downloaded with a single script.

## Download checkpoints from Yandex Disk

```
python scripts/download_checkpoints.py \
  --url "https://disk.yandex.ru/d/Y13vly-deqTA6Q" \
  --out-dir .
```

After running it, you should see the directory:
```
saved/
  hifigan_baseline/
    model_best.pth
    checkpoint-epoch20.pth
    checkpoint-epoch50.pth
    config.yaml
    ...
```

## Training

Training is performed via train.py and Hydra configs.

Example command for the final HiFi-GAN model:
```
python train.py --config-name hifigan_baseline
```

## Inference (synthesize.py)

The script synthesize.py is the main entrypoint for evaluation and TTS.
It supports three modes:
	1.	resynthesize – audio → mel → HiFi-GAN → audio.
	2.	tts_dataset – CustomDir dataset of texts → ESPNet Text2Speech → mel → HiFi-GAN → audio.
	3.	tts_single – single text string → ESPNet → mel → HiFi-GAN → audio.

Resynthesis mode

Resynthesize ground-truth audio (e.g. from LJSpeech) using the trained vocoder:
```
python synthesize.py \
  --checkpoint-path saved/hifigan_baseline/model_best.pth \
  --mode resynthesize \
  --input-audio-dir LJSpeech-1.1/wavs \
  --output-dir outputs/resynth_ljspeech \
  --device cuda
```
TTS from dataset (tts_dataset)

TTS evaluatio requires a CustomDirDataset format:
```
MyDataset/
└── transcriptions
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    ├── ...
    └── UtteranceIDn.txt
```
```
python synthesize.py \
  --checkpoint-path saved/hifigan_baseline/model_best.pth \
  --mode tts_dataset \
  --custom-dir /path/to/MyDataset \
  --output-dir outputs/mydataset_tts \
  --espnet_model espnet/kan-bayashi_ljspeech_vits \
  --device cuda
```

Single-text TTS (tts_single)

Generate audio from a single text prompt:
```
python synthesize.py \
  --checkpoint-path saved/hifigan_baseline/model_best.pth \
  --mode tts_single \
  --text "Dmitri Shostakovich was a Soviet-era Russian composer and pianist..." \
  --output-path outputs/shostakovich.wav \
  --espnet_model espnet/kan-bayashi_ljspeech_vits \
  --device cuda
```

The repository contains a Colab-ready demo notebook Demo.ipynb that demonstrates resynthesis on a sample subset of LJSpeech (mode=resynthesize), TTS on the external CustomDir dataset (mode=tts_dataset), synthesis of the 5 mandatory MOS sentences (mode=tts_single).

All experiments are logged to Comet ML (project dla_hw3_hifigan)

9. License

The project is distributed under the MIT License.
Copyright (c) 2024 Maxim Afanasev (olimpiadnik)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

