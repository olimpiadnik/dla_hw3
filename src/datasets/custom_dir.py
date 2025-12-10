import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):

    def __init__(self, root: str):
        super().__init__()

        self.root = Path(root)
        self.transcriptions_dir = self.root / "transcriptions"

        if not self.transcriptions_dir.exists():
            raise FileNotFoundError(
                f"transcriptions/ directory not found under {self.root}"
            )

        self.items: List[Tuple[str, Path]] = []
        for fname in sorted(os.listdir(self.transcriptions_dir)):
            if not fname.lower().endswith(".txt"):
                continue
            utt_id = os.path.splitext(fname)[0]
            path = self.transcriptions_dir / fname
            self.items.append((utt_id, path))

        if len(self.items) == 0:
            raise RuntimeError(
                f"No *.txt files found in {self.transcriptions_dir}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        utt_id, path = self.items[idx]

        with path.open("r", encoding="utf-8") as f:
            text = f.read().strip()

        return {
            "utt_id": utt_id,
            "text": text,
        }
