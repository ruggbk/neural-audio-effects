import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


WINDOW_SIZE = 4096
HOP_SIZE = 2048
SEED = 42


def _build_pairs(rendered_dir: Path, guitar_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for rendered_path in sorted(rendered_dir.glob("*.wav")):
        guitar_stem = rendered_path.stem + "_mix"
        guitar_path = guitar_dir / (guitar_stem + ".wav")
        if guitar_path.exists():
            pairs.append((guitar_path, rendered_path))
    return pairs


def _count_windows(path: Path) -> int:
    info = torchaudio.info(str(path))
    n_samples = info.num_frames
    if n_samples < WINDOW_SIZE:
        return 0
    return (n_samples - WINDOW_SIZE) // HOP_SIZE + 1


class AudioPairDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]]):
        self.index: list[tuple[Path, Path, int]] = []
        for guitar_path, rendered_path in pairs:
            n = _count_windows(guitar_path)
            for i in range(n):
                self.index.append((guitar_path, rendered_path, i))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        guitar_path, rendered_path, window_idx = self.index[idx]
        frame_offset = window_idx * HOP_SIZE

        guitar, _ = torchaudio.load(
            str(guitar_path), frame_offset=frame_offset, num_frames=WINDOW_SIZE
        )
        rendered, _ = torchaudio.load(
            str(rendered_path), frame_offset=frame_offset, num_frames=WINDOW_SIZE
        )

        return guitar[:1], rendered[:1]


def make_datasets(
    rendered_dir: Path,
    guitar_dir: Path,
    val_fraction: float = 0.1,
) -> tuple[AudioPairDataset, AudioPairDataset]:
    pairs = _build_pairs(rendered_dir, guitar_dir)
    rng = random.Random(SEED)
    rng.shuffle(pairs)

    n_val = max(1, int(len(pairs) * val_fraction))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    return AudioPairDataset(train_pairs), AudioPairDataset(val_pairs)
