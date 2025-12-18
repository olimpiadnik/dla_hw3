import argparse
from pathlib import Path
import zipfile

import yadisk


def download_and_unpack(public_url: str, out_dir: str, zip_name: str = "hifigan_hifigan_baseline_ep50.zip"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / zip_name

    y = yadisk.YaDisk()
    print(f"[download_checkpoints] Downloading from {public_url} to {zip_path} ...")
    # ВАЖНО: передаём строку пути
    y.download_public(public_url, str(zip_path))

    print(f"[download_checkpoints] Unzipping {zip_path} into {out_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    print("[download_checkpoints] Done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Download HiFi-GAN checkpoints from Yandex Disk")
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Public Yandex Disk link to the .zip archive (e.g. https://disk.yandex.ru/d/XXXXXXXXXXXXXX)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory to unpack the archive into (default: current directory)",
    )
    parser.add_argument(
        "--zip-name",
        type=str,
        default="hifigan_hifigan_baseline_ep50.zip",
        help="Filename to save the downloaded zip as",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_and_unpack(args.url, args.out_dir, zip_name=args.zip_name)
