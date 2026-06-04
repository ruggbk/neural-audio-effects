"""
Generates a chromatic sweep MIDI file for DDSP training data.
Produces long sustained notes across the full instrument range at multiple velocities.

Usage:
    python src/generate_sweep.py --output data/midi/sweep.mid
"""
import argparse
from pathlib import Path

import pretty_midi


NOTE_DURATION_S = 2.0
GAP_S = 0.25
MIDI_MIN = 38   # C2 = 36
MIDI_MAX = 88   # C7 = 96
VELOCITIES = [16, 32, 64, 80, 100, 127]


def generate_sweep(output_path: Path) -> None:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    t = 0.0
    for velocity in VELOCITIES:
        for pitch in range(MIDI_MIN, MIDI_MAX + 1):
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=t,
                end=t + NOTE_DURATION_S,
            )
            instrument.notes.append(note)
            t += NOTE_DURATION_S + GAP_S

    pm.instruments.append(instrument)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(output_path))
    total_minutes = t / 60
    print(f"Written to {output_path} ({total_minutes:.1f} minutes, {len(instrument.notes)} notes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/midi/sweep.mid")
    parser.add_argument("--min-pitch", type=int, default=MIDI_MIN)
    parser.add_argument("--max-pitch", type=int, default=MIDI_MAX)
    parser.add_argument("--note-duration", type=float, default=NOTE_DURATION_S)
    parser.add_argument("--gap", type=float, default=GAP_S)
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    generate_sweep(repo_root / args.output)
