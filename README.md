# neural-audio-effects
**Author:** Brandon Rugg


Experimental machine learning–based audio effect that learns timbral transformations from guitar input using virtual instruments as training targets.

## Approach

Rather than modeling an existing amp or pedal, this project generates training data by converting clean guitar recordings to MIDI and routing them through a virtual instrument (VSTi). A neural network is then trained on the resulting (guitar, VSTi output) pairs, learning a timbral transformation that captures some harmonic character of the target instrument without directly emulating it.

The trained model is exported as a VST plugin via [Neutone](https://neutone.space/), making it usable as a real-time effect in a DAW.

## Pipeline

1. **JAMS → MIDI** — GuitarSet's per-string note annotations are converted to MIDI files using `src/parse_jams.py`, avoiding the need for live pitch detection on polyphonic guitar audio.
2. **MIDI → VSTi audio** — MIDI files are batch-rendered through a VSTi chain in Reaper via `src/render_midi.py`, using the reapy Python bridge. Each experiment is defined by a config file in `configs/` and a corresponding Reaper template in `reaper/`.
3. **Training** — A TCN or Wave-U-Net is trained on (guitar, VSTi output) pairs using `src/train.py`. Architecture and hyperparameters are set per-experiment in the config file.
4. **Export** — The trained model is wrapped with the Neutone SDK for deployment as a real-time VST plugin.

## Data

Training data is sourced from [GuitarSet](https://guitarset.weebly.com/), a dataset of annotated guitar recordings recorded via direct input. The included note annotations are used to drive the VSTi directly.

## Samples

Two source clips from GuitarSet are used throughout:
- `00_BN1-129-Eb_comp` — bossa nova comping, Eb major, 129 BPM
- `00_Funk2-108-Eb_solo` — funk solo, Eb, 108 BPM

**Guitar input**

| Clip | File |
|---|---|
| Bossa nova comp | [guitarinput_00_BN1-129-Eb_comp_mix_guitar.mp3](samples/guitarinput_00_BN1-129-Eb_comp_mix_guitar.mp3) |
| Funk solo | [guitarinput_00_Funk2-108-Eb_solo_mix_guitar.mp3](samples/guitarinput_00_Funk2-108-Eb_solo_mix_guitar.mp3) |

**B3 Organ (target and model outputs)**

| | Bossa nova comp | Funk solo |
|---|---|---|
| VSTi render (target) | [b3organ_v2_target_00_BN1-129-Eb_comp.mp3](samples/b3organ_v2_target_00_BN1-129-Eb_comp.mp3) | [b3organ_v2_target_00_Funk2-108-Eb_solo.mp3](samples/b3organ_v2_target_00_Funk2-108-Eb_solo.mp3) |
| WaveUNet, epoch 100 | [b3organ_waveunet_poly_epoch_100_00_BN1-129-Eb_comp_mix.mp3](samples/b3organ_waveunet_poly_epoch_100_00_BN1-129-Eb_comp_mix.mp3) | [b3organ_waveunet_poly_epoch_100_00_Funk2-108-Eb_solo_mix.mp3](samples/b3organ_waveunet_poly_epoch_100_00_Funk2-108-Eb_solo_mix.mp3) |

**Harpsichord (target and model outputs)**

| | Bossa nova comp | Funk solo |
|---|---|---|
| VSTi render (target) | [harpsichord_target_00_BN1-129-Eb_comp.mp3](samples/harpsichord_target_00_BN1-129-Eb_comp.mp3) | [harpsichord_target_00_Funk2-108-Eb_solo.mp3](samples/harpsichord_target_00_Funk2-108-Eb_solo.mp3) |
| TCN mono, epoch 25 | [harpsichord_mono_epoch_025_00_BN1-129-Eb_comp_mix.mp3](samples/harpsichord_mono_epoch_025_00_BN1-129-Eb_comp_mix.mp3) | [harpsichord_mono_epoch_025_00_Funk2-108-Eb_solo_mix.mp3](samples/harpsichord_mono_epoch_025_00_Funk2-108-Eb_solo_mix.mp3) |
| TCN poly, epoch 25 | [harpsichord_poly_epoch_025_00_BN1-129-Eb_comp_mix.mp3](samples/harpsichord_poly_epoch_025_00_BN1-129-Eb_comp_mix.mp3) | [harpsichord_poly_epoch_025_00_Funk2-108-Eb_solo_mix.mp3](samples/harpsichord_poly_epoch_025_00_Funk2-108-Eb_solo_mix.mp3) |
| WaveUNet poly, epoch 25 | [harpsichord_waveunet_poly_epoch_025_00_BN1-129-Eb_comp_mix.mp3](samples/harpsichord_waveunet_poly_epoch_025_00_BN1-129-Eb_comp_mix.mp3) | [harpsichord_waveunet_poly_epoch_025_00_Funk2-108-Eb_solo_mix.mp3](samples/harpsichord_waveunet_poly_epoch_025_00_Funk2-108-Eb_solo_mix.mp3) |
| WaveUNet poly, epoch 100 | [harpsichord_waveunet_poly_epoch_100_00_BN1-129-Eb_comp_mix.mp3](samples/harpsichord_waveunet_poly_epoch_100_00_BN1-129-Eb_comp_mix.mp3) | [harpsichord_waveunet_poly_epoch_100_00_Funk2-108-Eb_solo_mix.mp3](samples/harpsichord_waveunet_poly_epoch_100_00_Funk2-108-Eb_solo_mix.mp3) |

## Experiments

| Name | VSTi | Architecture | Data |
|---|---|---|---|
| b3_organ_v2 | Hammond B3 (Vintage Organs via Kontakt) | TCN 32ch | Polyphonic |
| harpsichord_poly_v3 | Harpsichord (Kontakt Factory Library) | TCN 64ch | Polyphonic |
| harpsichord_mono_v3 | Harpsichord (Kontakt Factory Library) | TCN 64ch | Monophonic only |
| harpsichord_waveunet_poly | Harpsichord (Kontakt Factory Library) | Wave-U-Net | Polyphonic |
| b3_organ_waveunet_poly | Hammond B3 (Vintage Organs via Kontakt) | Wave-U-Net | Polyphonic |

## Setup

**Prerequisites**: [Reaper](https://www.reaper.fm/) and the [reapy bridge](https://python-reapy.readthedocs.io/en/latest/install_guide.html) must be configured before running the rendering step.

1. Install dependencies: `conda env create -f environment.yml`
2. Download GuitarSet annotations and mono pickup mix audio to `data/guitarset/`
3. Open the relevant Reaper template from `reaper/` with your VSTi configured
4. Run `python src/parse_jams.py` to generate MIDI, then `python src/render_midi.py --config configs/<name>.yaml` to render VSTi audio
5. Run `python src/train.py --config configs/<name>.yaml` to train the model (GPU recommended)
6. Run `python src/neutone_wrapper.py --config configs/<name>.yaml` to export the trained model as a Neutone VST plugin

## Status

Pipeline complete. Five experiments concluded across TCN and Wave-U-Net architectures on Hammond B3 and Harpsichord targets. All models exported as Neutone VST plugins.
