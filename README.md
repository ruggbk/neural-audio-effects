# neural-audio-effects

Experimental machine learning–based audio effect that learns timbral transformations from guitar input using virtual instruments as training targets.

## Approach

Rather than modeling an existing amp or pedal, this project generates training data by converting clean guitar recordings to MIDI and routing them through a virtual instrument (VSTi). A neural network is then trained on the resulting (guitar, VSTi output) pairs, learning a timbral transformation that captures some harmonic character of the target instrument without directly emulating it.

The trained model is exported as a VST plugin via [Neutone](https://neutone.space/), making it usable as a real-time effect in a DAW.

## Pipeline

1. **JAMS → MIDI** — GuitarSet's per-string note annotations are converted to MIDI files using `src/parse_jams.py`, avoiding the need for live pitch detection on polyphonic guitar audio.
2. **MIDI → VSTi audio** — MIDI files are batch-rendered through a VSTi chain in Reaper via `src/render_midi.py`, using the reapy Python bridge. Each experiment is defined by a config file in `configs/` and a corresponding Reaper template in `reaper/`.
3. **Training** — A TCN (Temporal Convolutional Network) is trained on (guitar, VSTi output) pairs using `src/train.py`.
4. **Export** — The trained model is wrapped with the Neutone SDK for deployment as a real-time VST plugin.

## Data

Training data is sourced from [GuitarSet](https://guitarset.weebly.com/), a dataset of annotated guitar recordings recorded via direct input. The included note annotations are used to drive the VSTi directly.

## Samples

`00_BN1-129-Eb_comp` — bossa nova comping, Eb major, 129 BPM

| | File |
|---|---|
| Guitar (input) | [00_BN1-129-Eb_comp_mix.wav](samples/00_BN1-129-Eb_comp_mix.wav) |
| B3 organ render (target) | [00_BN1-129-Eb_comp.wav](samples/00_BN1-129-Eb_comp.wav) |

## Experiments

| Name | VSTi | Status |
|---|---|---|
| b3_organ | Hammond B3 (Vintage Organs via Kontakt) | Rendering |

## Setup

**Prerequisites**: [Reaper](https://www.reaper.fm/) and the [reapy bridge](https://python-reapy.readthedocs.io/en/latest/install_guide.html) must be configured before running the rendering step.

1. Install dependencies: `conda env create -f environment.yml`
2. Download GuitarSet annotations and mono pickup mix audio to `data/guitarset/`
3. Open the relevant Reaper template from `reaper/` with your VSTi configured
4. Run `notebooks/02_pipeline.ipynb` to generate MIDI and render VSTi audio

## Status

In progress — pipeline complete, training in development.
