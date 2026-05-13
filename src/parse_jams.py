import jams
import numpy as np
import pretty_midi
import soundfile as sf
from pathlib import Path


VELOCITY_MIN = 40
VELOCITY_MAX = 120
RMS_WINDOW_S = 0.02  # 20ms window after note onset for energy measurement


def _note_rms(audio: np.ndarray, sr: int, onset: float) -> float:
    start = int(onset * sr)
    end = start + int(RMS_WINDOW_S * sr)
    window = audio[start:min(end, len(audio))]
    if len(window) == 0:
        return 0.0
    return float(np.sqrt(np.mean(window ** 2)))


def _rms_to_velocities(rms_values: list) -> list:
    """Map RMS values to MIDI velocities using log scaling, normalized per file."""
    arr = np.array(rms_values, dtype=np.float32)
    arr = np.clip(arr, 1e-8, None)
    log_rms = np.log(arr)
    lo, hi = log_rms.min(), log_rms.max()
    if hi - lo < 1e-6:
        normalized = np.full_like(log_rms, 0.5)
    else:
        normalized = (log_rms - lo) / (hi - lo)
    velocities = VELOCITY_MIN + normalized * (VELOCITY_MAX - VELOCITY_MIN)
    return [int(round(v)) for v in velocities]


def parse_jams_to_midi(jams_path: Path, output_path: Path, guitar_dir: Path) -> None:
    jam = jams.load(str(jams_path))

    guitar_path = guitar_dir / (jams_path.stem + '_mix.wav')
    if guitar_path.exists():
        audio, sr = sf.read(str(guitar_path), always_2d=True)
        audio_mono = audio[:, 0]
        use_velocity = True
    else:
        use_velocity = False

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    notes = []
    rms_values = []

    for track in jam.search(namespace='note_midi'):
        for obs in track.data:
            onset = obs.time
            duration = obs.duration
            offset = onset + duration
            pitch = int(round(obs.value))
            notes.append((onset, offset, pitch))
            if use_velocity:
                rms_values.append(_note_rms(audio_mono, sr, onset))

    velocities = _rms_to_velocities(rms_values) if use_velocity and rms_values else [80] * len(notes)

    for (onset, offset, pitch), velocity in zip(notes, velocities):
        instrument.notes.append(pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=onset,
            end=offset,
        ))

    instrument.notes.sort(key=lambda n: n.start)
    midi.instruments.append(instrument)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))


def convert_all(annotation_dir: Path, output_dir: Path, guitar_dir: Path) -> None:
    annotation_files = sorted(annotation_dir.glob('*.jams'))
    output_dir.mkdir(parents=True, exist_ok=True)

    for jams_path in annotation_files:
        output_path = output_dir / (jams_path.stem + '.mid')
        parse_jams_to_midi(jams_path, output_path, guitar_dir)
        print(f'Converted {jams_path.name}')

    print(f'Done. {len(list(output_dir.glob("*.mid")))} MIDI files in {output_dir}')


if __name__ == '__main__':
    repo_root = Path(__file__).parent.parent
    convert_all(
        annotation_dir=repo_root / 'data' / 'guitarset' / 'annotation',
        output_dir=repo_root / 'data' / 'midi',
        guitar_dir=repo_root / 'data' / 'guitarset' / 'audio',
    )
