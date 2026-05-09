import jams
import pretty_midi
from pathlib import Path


VELOCITY = 80


def parse_jams_to_midi(jams_path: Path, output_path: Path) -> None:
    jam = jams.load(str(jams_path))
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for track in jam.search(namespace='note_midi'):
        for obs in track.data:
            onset = obs.time
            offset = onset + obs.duration
            pitch = int(round(obs.value))
            note = pretty_midi.Note(
                velocity=VELOCITY,
                pitch=pitch,
                start=onset,
                end=offset,
            )
            instrument.notes.append(note)

    instrument.notes.sort(key=lambda n: n.start)
    midi.instruments.append(instrument)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))


def process_all(annotation_dir: Path, output_dir: Path) -> None:
    annotation_files = sorted(annotation_dir.glob('*.jams'))
    output_dir.mkdir(parents=True, exist_ok=True)

    for jams_path in annotation_files:
        output_path = output_dir / (jams_path.stem + '.mid')
        if output_path.exists():
            continue
        parse_jams_to_midi(jams_path, output_path)
        print(f'Converted {jams_path.name}')

    print(f'Done. {len(list(output_dir.glob("*.mid")))} MIDI files in {output_dir}')


if __name__ == '__main__':
    repo_root = Path(__file__).parent.parent
    process_all(
        annotation_dir=repo_root / 'data' / 'guitarset' / 'annotation',
        output_dir=repo_root / 'data' / 'midi',
    )
