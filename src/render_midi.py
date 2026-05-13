import tempfile
import time
from pathlib import Path

import pretty_midi
import reapy
from reapy import reascript_api as RPR
import yaml


# Allows reverb/release to decay before the render ends
TAIL_SECONDS = 1.0
# Reaper action: "Render project, using the most recent render settings"
RENDER_ACTION = 42230


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_vsti_track(project: reapy.Project) -> reapy.Track:
    assert len(project.tracks) > 0, "No tracks found — is the template project open in Reaper?"
    return project.tracks[0]


def _prepare_midi(midi_path: Path, duration_scale: float, min_duration_s: float, onset_offset_s: float) -> Path:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    for instrument in pm.instruments:
        for note in instrument.notes:
            new_dur = max((note.end - note.start) * duration_scale, min_duration_s)
            note.start = max(0.0, note.start - onset_offset_s)
            note.end = note.start + new_dur
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    pm.write(tmp.name)
    tmp.close()
    return Path(tmp.name)


def render_clip(
    project: reapy.Project,
    track: reapy.Track,
    midi_path: Path,
    output_path: Path,
    config: dict,
) -> None:
    """Render a single MIDI file through the VSTi track and write a WAV to output_path.

    Clears any existing media items from the track before inserting the new MIDI clip.
    Blocks until the output file exists or a deadline is reached.
    """
    duration_scale = config.get("duration_scale", 1.0)
    min_duration_s = config.get("min_duration_s", 0.05)
    onset_offset_s = config.get("onset_offset_s", 0.0)
    insert_path = _prepare_midi(midi_path, duration_scale, min_duration_s, onset_offset_s)

    while RPR.GetTrackNumMediaItems(track.id) > 0:
        item = RPR.GetTrackMediaItem(track.id, 0)
        RPR.DeleteTrackMediaItem(track.id, item)

    RPR.SetOnlyTrackSelected(track.id)
    project.cursor_position = 0
    RPR.InsertMedia(str(insert_path), 0)

    pm = pretty_midi.PrettyMIDI(str(insert_path))
    duration = pm.get_end_time() + TAIL_SECONDS

    project.time_selection = (0, duration)

    pid = project.id
    RPR.GetSetProjectInfo(pid, "RENDER_REALTIME", 1, True)
    RPR.GetSetProjectInfo_String(pid, "RENDER_FILE", str(output_path.parent), True)
    RPR.GetSetProjectInfo_String(pid, "RENDER_PATTERN", output_path.stem, True)
    RPR.GetSetProjectInfo(pid, "RENDER_SRATE", config["sample_rate"], True)
    RPR.GetSetProjectInfo(pid, "RENDER_BOUNDSFLAG", 2, True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    RPR.Main_OnCommand(RENDER_ACTION, 0)

    deadline = time.time() + duration + 10
    while not output_path.exists() and time.time() < deadline:
        time.sleep(0.5)


def render_all(midi_dir: Path, config: dict, repo_root: Path) -> None:
    midi_files = sorted(midi_dir.glob("*.mid"))
    output_dir = repo_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    project = reapy.Project()
    track = get_vsti_track(project)

    for midi_path in midi_files:
        output_path = output_dir / (midi_path.stem + ".wav")
        if output_path.exists():
            print(f"Skipping {midi_path.name}")
            continue
        render_clip(project, track, midi_path, output_path, config)
        print(f"Rendered {midi_path.name}")

    print(f"Done. {len(list(output_dir.glob('*.wav')))} files in {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b3_organ.yaml")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    config = load_config(repo_root / args.config)
    render_all(
        midi_dir=repo_root / "data" / "midi",
        config=config,
        repo_root=repo_root,
    )
