import random
from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi
import soundfile as sf
import torch
from torch.utils.data import Dataset


WINDOW_SIZE = 4096
HOP_SIZE = 2048
SEED = 42
SAMPLE_RATE = 44100
MIN_RMS = 1e-3  # ~-60dB; windows below this are excluded as silence


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


def _is_monophonic_window(notes: list, t_start: float, t_end: float) -> bool:
    active = [n for n in notes if n.start < t_end and n.end > t_start]
    if len(active) <= 1:
        return True
    active.sort(key=lambda n: n.start)
    for i in range(len(active) - 1):
        if active[i].end > active[i + 1].start:
            return False
    return True


class AudioPairDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        midi_dir: Optional[Path] = None,
        monophonic_only: bool = False,
    ):
        self.index: list[tuple[Path, Path, int]] = []
        total_before = 0
        silence_removed = 0
        for guitar_path, rendered_path in pairs:
            n = _count_windows(guitar_path, rendered_path)
            total_before += n

            guitar_audio, _ = sf.read(str(guitar_path), always_2d=True)
            guitar_mono = guitar_audio[:, 0]

            notes = None
            if monophonic_only and midi_dir is not None:
                midi_path = midi_dir / (rendered_path.stem + ".mid")
                if midi_path.exists():
                    pm = pretty_midi.PrettyMIDI(str(midi_path))
                    notes = [note for inst in pm.instruments for note in inst.notes]

            for i in range(n):
                start = i * HOP_SIZE
                chunk = guitar_mono[start:start + WINDOW_SIZE]
                if float(np.sqrt(np.mean(chunk ** 2))) < MIN_RMS:
                    silence_removed += 1
                    continue
                if notes is not None:
                    t_start = start / SAMPLE_RATE
                    t_end = (start + WINDOW_SIZE) / SAMPLE_RATE
                    if not _is_monophonic_window(notes, t_start, t_end):
                        continue
                self.index.append((guitar_path, rendered_path, i))

        removed = []
        if silence_removed:
            removed.append(f"silence: {silence_removed:,}")
        if monophonic_only:
            poly_removed = total_before - silence_removed - len(self.index)
            removed.append(f"polyphonic: {poly_removed:,}")
        if removed and total_before > 0:
            pct = len(self.index) / total_before * 100
            print(f"Windows: {total_before:,} → {len(self.index):,} ({', '.join(removed)} removed, {pct:.1f}% kept)")

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
    midi_dir: Optional[Path] = None,
    monophonic_only: bool = False,
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

    return (
        AudioPairDataset(train_pairs, midi_dir=midi_dir, monophonic_only=monophonic_only),
        AudioPairDataset(val_pairs, midi_dir=midi_dir, monophonic_only=monophonic_only),
    )
