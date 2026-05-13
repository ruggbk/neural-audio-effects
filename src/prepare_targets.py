"""
Post-rendering envelope matching step. Transfers the guitar's amplitude envelope
onto the organ renders so the training targets follow guitar dynamics instead of
organ sustain. Run after render_midi.py, before train.py.

Usage:
    python src/prepare_targets.py --config configs/b3_organ_v2.yaml

Output goes to <output_dir>_env/ (e.g. data/rendered/b3_organ_v2_env/).
To train on envelope-matched targets, update output_dir in your config to point there.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


ENV_WINDOW_S = 0.02  # 20ms RMS window for envelope estimation
ENV_HOP_S = 0.010    # 10ms hop between windows
NOISE_FLOOR = 1e-4   # prevents division by near-zero organ envelope
MAX_GAIN = 10.0      # caps gain to ~20dB to avoid amplifying noise in silences


def _envelope(audio: np.ndarray, sr: int) -> np.ndarray:
    window = int(ENV_WINDOW_S * sr)
    hop = int(ENV_HOP_S * sr)
    centers, rms_values = [], []
    for i in range(0, len(audio) - window, hop):
        rms_values.append(float(np.sqrt(np.mean(audio[i:i + window] ** 2))))
        centers.append(i + window // 2)
    if not centers:
        return np.zeros(len(audio))
    return np.interp(np.arange(len(audio)), centers, rms_values)


def match_envelope(guitar_path: Path, organ_path: Path, output_path: Path) -> None:
    guitar, sr = sf.read(str(guitar_path), always_2d=True)
    organ, _ = sf.read(str(organ_path), always_2d=True)

    guitar_mono = guitar[:, 0]
    organ_mono = organ[:, 0]

    n = min(len(guitar_mono), len(organ_mono))
    guitar_mono = guitar_mono[:n]
    organ_mono = organ_mono[:n]

    gain = _envelope(guitar_mono, sr) / (_envelope(organ_mono, sr) + NOISE_FLOOR)
    gain = np.clip(gain, 0.0, MAX_GAIN)

    matched = np.clip(organ_mono * gain, -1.0, 1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), matched, sr, subtype="FLOAT")


def prepare_all(config: dict, repo_root: Path) -> None:
    rendered_dir = repo_root / config["output_dir"]
    guitar_dir = repo_root / "data" / "guitarset" / "audio"
    output_dir = rendered_dir.parent / (rendered_dir.name + "_env")
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered_files = sorted(rendered_dir.glob("*.wav"))
    count = 0
    for rendered_path in rendered_files:
        guitar_path = guitar_dir / (rendered_path.stem + "_mix.wav")
        if not guitar_path.exists():
            print(f"No guitar match for {rendered_path.name}, skipping")
            continue
        match_envelope(guitar_path, rendered_path, output_dir / rendered_path.name)
        count += 1
        print(f"Matched {rendered_path.name}")

    print(f"Done. {count} files written to {output_dir}")
    print(f"To train on these, set output_dir: {output_dir.relative_to(repo_root)} in your config.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    config = yaml.safe_load(open(repo_root / args.config))
    prepare_all(config, repo_root)
