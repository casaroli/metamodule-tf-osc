---
name: tfexample-demos
description: Regenerate the 30-WAV demo pack that tours every TFOsc control (anchor timbres, morph sweeps, tuning, LFO, warp, composite figures). Use when the user wants fresh audio samples or has changed the model / DSP and wants to hear the result.
---

# Regenerate TFOsc demo pack

Renders 30 short WAV files under `/tmp/tfosc_samples/` (or another
directory if passed as an argument). Listens via `afplay` on macOS.

## Prerequisites

1. Headless simulator built. Use `tfexample-build-sim headless` skill if not.
2. Training venv at `models/train/.venv/` for numpy/scipy.

## Run

```bash
cd tests
rm -rf /tmp/tfosc_samples
../models/train/.venv/bin/python3 generate_samples.py /tmp/tfosc_samples
```

Takes ~30-60 s depending on host. Output files prefixed by group:

| Range | Group | Content |
|---|---|---|
| `01–09` | anchors | Each of the 9 SIREN anchor timbres at 200 Hz, 2 s each |
| `10–15` | morph sweeps | Bottom edge, mid edge, top edge, diagonals, spiral |
| `20–21` | tuning | Pitch sweep, fine-tune detune sweep |
| `30–34` | LFO | Slow/shallow, slow/deep, medium, fast, rate sweep |
| `40–42` | warp | PD on sine, on saw, PWM-style wobble |
| `50–55` | composite | Pentatonic figures, evolving pad, neural lead, bell hit |

## Listen

```bash
for f in /tmp/tfosc_samples/*.wav; do
  echo "▸ $(basename $f)"
  afplay "$f"
done
```

## Add a new demo

Edit `tests/generate_samples.py`. The helpers are:

- `render_seg(dur_s, **knobs)` — render one fixed-knob segment
- `sweep(steps, step_dur=0.20)` — concatenate per-step renders to fake
  knob automation (since the headless sim can't vary params mid-render)
- `crossfade(a, b, ms=30)` — for legato joins between segments
- `save(path, arr)` — write a WAV (handles normalization + clipping)

Knob names available in `**knobs`: `freq, fine, morph_x, morph_y,
lfo_rate, lfo_depth, warp, fm_amt`. Defaults are sensible (no LFO,
no warp, no FM).

## If samples come out silent

Same triage as `tfexample-test`: see that skill's "silent" section.
The headless render path is identical; if pytest passes, generate_samples
should too.

## Output file format

The headless sim writes WAV (RIFF/WAVE float32 stereo, 48 kHz) despite
the `.raw` extension. `generate_samples.py` parses the WAV header,
reads the floats, applies gain normalization, then writes a clean
int16-PCM WAV via scipy.io.wavfile.
