# neural-audio-effects

Experimental machine learning–based audio effect that learns timbral transformations from guitar input using virtual instruments as training targets.

## Approach

Rather than modeling an existing amp or pedal, this project generates training data by converting clean guitar recordings to MIDI and routing them through a virtual instrument (VSTi). A neural network is then trained on the resulting (guitar, VSTi output) pairs, learning a timbral transformation that captures some harmonic character of the target instrument without directly emulating it.

The trained model is exported as a VST plugin via [Neutone](https://neutone.space/), making it usable as a real-time effect in a DAW.

## Data

Training data is sourced from [GuitarSet](https://guitarset.weebly.com/), a dataset of annotated guitar recordings. The included note annotations are used to drive the VSTi directly, avoiding the need for live pitch detection.

## Status

In progress.
