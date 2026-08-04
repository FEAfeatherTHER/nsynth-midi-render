# NSynth MIDI Render

Render one single-instrument MIDI file with one NSynth timbre parquet.

## Python API

```python
from nsynth_midi_render import NoteSynthesizer

synth = NoteSynthesizer(
    "nsynth/keyboard/keyboard_electronic_037.parquet",
    attack_ms=0,
    release_ms=500,
    random_ratio=0.1,
    seed=42,
    normalize=False,
)

audio = synth.render_audio("input.mid")
output = synth.render("input.mid", "output/input.wav")
```

`render_audio()` returns a mono `float32` NumPy array. `render()` writes a
FLOAT WAV and returns its path. If the timbre does not contain every MIDI
pitch, both methods skip the complete MIDI and return `None`.

## Command line

```bash
python render_midi.py \
  --timbre nsynth/keyboard/keyboard_electronic_037.parquet \
  --midi midi/keyboard/Rock_3142.mid \
  --output rendered/Rock_3142.wav
```

Use `python render_midi.py --help` for all parameters. Exit code `0` means
success, `1` means invalid input, and `2` means the timbre was missing a
required pitch.

## Audio behavior

- NSynth audio before 3 seconds is the attack/sustain region.
- MIDI notes longer than 3 seconds loop the 2--3 second region with a 50 ms
  crossfade.
- At MIDI Note Off, audio crossfades into the original NSynth release region
  after 3 seconds.
- `release_ms` limits the original release length. It does not extend release
  beyond the audio available in the parquet (normally about 1 second).
- `attack_ms` defaults to `0` because NSynth already contains a natural attack.
- `normalize` defaults to `False`. FLOAT WAV output preserves peaks above 1
  instead of silently clipping them.

MIDI format 0 and 1 are accepted when exactly one non-drum instrument contains
notes. MIDI format 2 is rejected. Control changes, sustain pedal, and pitch bend
are reported but ignored.

## Parquet requirements

Each row must contain `pitch`, `velocity`, `sample_rate`, and `audio`. `audio`
must contain a mono array and `sampling_rate`. All rows must use one sample rate
and each `(pitch, velocity)` pair must be unique.

MIDI velocity is matched to the nearest velocity available for that pitch.

## Dependencies

```bash
pip install -r requirements.txt
```
