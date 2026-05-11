import random
from pathlib import Path

import soundfile as sf
import torch
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


def _count_windows(guitar_path: Path, rendered_path: Path) -> int:
    n_samples = min(sf.info(str(guitar_path)).frames, sf.info(str(rendered_path)).frames)
    if n_samples < WINDOW_SIZE:
        return 0
    return (n_samples - WINDOW_SIZE) // HOP_SIZE + 1


class AudioPairDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]]):
        self.index: list[tuple[Path, Path, int]] = []
        for guitar_path, rendered_path in pairs:
            n = _count_windows(guitar_path, rendered_path)
            for i in range(n):
                self.index.append((guitar_path, rendered_path, i))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        guitar_path, rendered_path, window_idx = self.index[idx]
        start = window_idx * HOP_SIZE

        # always_2d=True ensures consistent (frames, channels) shape for mono and stereo files
        guitar, _ = sf.read(str(guitar_path), start=start, stop=start + WINDOW_SIZE, always_2d=True)
        rendered, _ = sf.read(str(rendered_path), start=start, stop=start + WINDOW_SIZE, always_2d=True)

        guitar_t = torch.from_numpy(guitar.T[:1]).float()
        rendered_t = torch.from_numpy(rendered.T[:1]).float()

        return guitar_t, rendered_t


def make_datasets(
    rendered_dir: Path,
    guitar_dir: Path,
    val_fraction: float = 0.1,
) -> tuple[AudioPairDataset, AudioPairDataset]:
    """Build train and validation datasets from matched guitar/rendered audio pairs.

    Pairs are shuffled with a fixed seed before splitting so the split is
    deterministic but not alphabetically ordered by filename.

    Returns (train_dataset, val_dataset).
    """
    pairs = _build_pairs(rendered_dir, guitar_dir)
    rng = random.Random(SEED)
    rng.shuffle(pairs)

    n_val = max(1, int(len(pairs) * val_fraction))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    return AudioPairDataset(train_pairs), AudioPairDataset(val_pairs)
