"""
Inspect hex pickup audio and JAMS per-string annotations for one track.
Run this to verify structure before building hex data into the training pipeline.

Usage:
    python src/inspect_hex.py
"""
from pathlib import Path

import jams
import numpy as np
import soundfile as sf


WINDOW_SIZE = 4096
HOP_SIZE = 2048
SAMPLE_RATE = 44100
MIN_RMS = 1e-3
MIN_STRING_ACTIVITY_S = 2.0  # skip strings with less than this many seconds of notes


def get_string_notes(jam: jams.JAMS) -> list[list]:
    """Extract per-string note lists from a JAMS file. Returns list of 6 note lists."""
    tracks = jam.search(namespace='note_midi')
    return [list(track.data) for track in tracks]


def string_activity_s(notes) -> float:
    """Total duration of notes on a string in seconds."""
    return sum(obs.duration for obs in notes)


def is_string_active_window(notes, t_start: float, t_end: float) -> bool:
    """Check if a string has any note activity in a time window."""
    return any(obs.time < t_end and (obs.time + obs.duration) > t_start for obs in notes)


def inspect(hex_path: Path, jams_path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Hex file: {hex_path.name}")
    print(f"JAMS file: {jams_path.name}")

    info = sf.info(str(hex_path))
    print(f"\nAudio: {info.channels} channels, {info.samplerate}Hz, {info.duration:.1f}s")

    jam = jams.load(str(jams_path))
    string_notes = get_string_notes(jam)
    print(f"\nPer-string activity:")
    string_names = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']
    for i, (name, notes) in enumerate(zip(string_names, string_notes)):
        activity = string_activity_s(notes)
        status = "SKIP" if activity < MIN_STRING_ACTIVITY_S else "keep"
        print(f"  String {i} ({name}): {len(notes):3d} notes, {activity:.1f}s total [{status}]")

    audio, sr = sf.read(str(hex_path), always_2d=True)
    n_samples = audio.shape[0]
    n_windows = (n_samples - WINDOW_SIZE) // HOP_SIZE + 1

    print(f"\nWindow filtering simulation ({n_windows} total windows):")
    for str_idx, (name, notes) in enumerate(zip(string_names, string_notes)):
        if string_activity_s(notes) < MIN_STRING_ACTIVITY_S:
            print(f"  String {str_idx} ({name}): skipped (low activity)")
            continue

        channel = audio[:, str_idx]
        kept = 0
        for i in range(n_windows):
            start = i * HOP_SIZE
            chunk = channel[start:start + WINDOW_SIZE]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < MIN_RMS:
                continue
            t_start = start / SAMPLE_RATE
            t_end = (start + WINDOW_SIZE) / SAMPLE_RATE
            if is_string_active_window(notes, t_start, t_end):
                kept += 1
        print(f"  String {str_idx} ({name}): {kept:,} windows kept")


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    hex_dir = repo_root / "data" / "guitarset" / "audio_hex-pickup_debleeded"
    jams_dir = repo_root / "data" / "guitarset" / "annotation"

    hex_files = sorted(hex_dir.glob("*.wav"))
    if not hex_files:
        print(f"No hex files found in {hex_dir}")
        print("Check the directory name matches your download.")
    else:
        print(f"Found {len(hex_files)} hex files.")
        # Inspect just the first file as a sanity check
        hex_path = hex_files[0]
        jams_stem = hex_path.stem.replace("_hex_cln", "").replace("_hex", "")
        jams_path = jams_dir / (jams_stem + ".jams")
        if not jams_path.exists():
            print(f"Could not find JAMS file: {jams_path}")
            print(f"Hex stem: {hex_path.stem}, tried: {jams_stem}")
        else:
            inspect(hex_path, jams_path)
