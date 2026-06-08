# TFExample — TensorFlow Lite Micro inside a MetaModule plugin

> ## ⚠️ EARLY EXPERIMENT — NOT TESTED ON REAL HARDWARE
>
> This is a **rough proof-of-concept**, not a shippable plugin. It works
> end-to-end **in the MetaModule simulator on a host computer**. It has
> **never been loaded onto a real MetaModule device**, and the central
> open question — whether per-sample TensorFlow Lite Micro inference makes
> audio budget at 48 kHz on the actual Cortex-A7 — is unanswered.
>
> Treat this as an **investigation log + working baseline** to build on.
> Plenty of obvious next steps remain unexplored: hardware validation,
> richer models, int8 quantization, training on real recorded waveforms,
> upstreaming the simulator patches, custom faceplate art, polyphony,
> and more. See [docs/FUTURE_WORK.md](docs/FUTURE_WORK.md) for the full
> list of directions worth pursuing.
>
> Use it as a starting point, not as a finished tool.

## What this is

The result is **TFOsc**, a neural wavetable-morph oscillator. A small SIREN
(sinusoidal-activation MLP) takes `(phase, morph_x, morph_y)` and returns one
audio sample. It runs inside the plugin's `update()` callback, called per
sample by the MetaModule audio engine. Users get a 2D timbral plane that
smoothly travels between nine learned anchor waveforms.

**Why this exists**: to answer "*can you usefully run a TensorFlow Lite
Micro model at audio rate inside a Eurorack plugin running on an embedded
ARM core?*" The answer so far is *yes inside the simulator, with unknown
headroom on the real hardware*. This repo captures everything learned in
the process so the next person doesn't start from zero.

## What works today

- **TFLite Micro vendored as a static lib** inside the plugin (`third_party/tflite-micro/`), built with the SDK's `arm-none-eabi-gcc 12.3` toolchain for Cortex-A7 + NEON.
- **9-anchor SIREN model** (~41 KB float `.tflite`), `3 → 96 → 96 → 1` with `sin(ω·Wx)` activations. Trains in ~30 s on a laptop.
- **8 knobs + 5 input jacks + 1 output jack**:
  Freq · Fine · Morph X · Morph Y · Warp · FM Amount · LFO Rate · LFO Depth · V/Oct · Sync · Morph X CV · Morph Y CV · FM In · Out.
- **Built-in auto-morph LFO**, **phase-distortion warp**, and **FM input** for phase mangling.
- **End-to-end automated audio tests** (13 of them) via a pytest harness that drives the MetaModule headless simulator and asserts on level, pitch, and spectral content.
- **Rich demo pack** — `python3 tests/generate_samples.py` renders 30 WAVs that tour every control.
- **Runs in the MetaModule simulator** (both GUI and headless modes). Hardware has not been tested yet.

## What this is NOT

- Not validated on real MetaModule hardware. Per-sample TFLM `Invoke()` at 48 kHz on Cortex-A7 may need `kInferenceStride > 1` + linear interpolation; the code already supports this knob, but it has not been measured. See [docs/FUTURE_WORK.md](docs/FUTURE_WORK.md).
- Not a great-sounding synth yet — the SIREN morph has clearly audible character, but the demo pack will tell you whether you find it musically useful.
- Not upstream-merged. The MetaModule headless simulator needed patches to link without ThorVG/FontStash and to register external plugins via `load_ext_builtin_plugins()`. Those changes live in a local clone of `4ms/metamodule`; see [docs/SIMULATOR_PATCHES.md](docs/SIMULATOR_PATCHES.md).
- Not a TensorFlow Lite kernel showcase. We only use **FullyConnected** and **Sin** kernels. Any extension to bigger models (LSTM amp modeling, RAVE-style decoders) is unexplored ground.

## Quick start

```bash
# 1. Build the hardware plugin (the .mmplugin you'd load on real hardware)
cd TFExample
cmake --fresh -B build -G Ninja \
  -DTOOLCHAIN_BASE_DIR=/Applications/ArmGNUToolchain/12.3.rel1/arm-none-eabi/bin
cmake --build build
# → ../metamodule-plugins/TFExample.mmplugin

# 2. (Optional) retrain the SIREN model
cd models/train
python3 -m venv .venv && .venv/bin/pip install tensorflow scipy pytest
.venv/bin/python3 train_siren_morph9.py
# regenerate the embedded C array
cd .. && xxd -i -n g_wavetable_morph_model_data wavetable_morph.tflite \
  | sed '...'  # see scripts/embed_model.sh or use the Claude skill

# 3. Run the audio test suite (requires the MetaModule simulator built locally)
cd ../../tests
.venv/bin/pytest

# 4. Generate the demo WAV pack
.venv/bin/python3 generate_samples.py /tmp/tfosc_samples
for f in /tmp/tfosc_samples/*.wav; do echo "▸ $f"; afplay "$f"; done
```

For the full workflow (sim build, GUI vs headless, regenerating model and demos), see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Repository layout

```
TFExample/
├── README.md                # ← you are here
├── CLAUDE.md                # context for Claude Code sessions
├── CMakeLists.txt           # plugin + tflite_micro static lib
├── plugin.cc                # init entry points (init / init_TFExample)
├── plugin.json              # MetaModule plugin metadata
├── plugin-mm.json           # 4ms maintainer metadata
├── tf_osc.{hh,cc}           # the TFOsc CoreProcessor implementation
├── tf_osc_elements.cc       # knob/jack/LED panel layout + register_module
├── docs/
│   ├── ARCHITECTURE.md      # why it's shaped this way
│   ├── DEVELOPMENT.md       # build/train/test/render workflow
│   ├── SIMULATOR_PATCHES.md # changes we made to 4ms/metamodule
│   └── FUTURE_WORK.md       # directions left unexplored
├── models/
│   ├── wavetable_morph.tflite               # current trained model
│   ├── wavetable_morph_model_data.{cc,h}    # embedded C array
│   ├── hello_world_*                        # original 3 KB sine demo (kept for history)
│   └── train/
│       ├── train_siren_morph9.py            # current trainer (9 anchors)
│       ├── train_siren_morph.py             # earlier 4-anchor trainer
│       └── train_wavetable_morph.py         # earlier ReLU-MLP attempt
├── tests/
│   ├── conftest.py                          # render() + tfosc_patch() helpers
│   ├── test_tfosc.py                        # 13 audio assertions
│   └── generate_samples.py                  # 30-WAV demo generator
├── third_party/
│   └── tflite-micro/                        # vendored slice (~5.3 MB)
└── assets/
    ├── tf_osc.png                           # faceplate (placeholder)
    └── components/                          # knob/jack/LED PNGs
```

## Architecture sketch

```
              Knobs / Jacks
                   │
                   ▼
   set_param(...) / set_input(...) cache values in members
                   │
        update() called per-sample by the audio thread
                   │
       ┌───────────┴───────────┐
       │ pitch / phase math    │   Freq · Fine · V/Oct → pitch_hz
       │ (classical DSP)       │   phase_ += pitch_hz / sr
       └───────────┬───────────┘
                   │
       ┌───────────┴───────────┐
       │ phase mangling        │   FM In × FM Amt added to phase
       │ (Warp + FM)           │   Casio-PD piecewise-linear warp
       └───────────┬───────────┘
                   │
       ┌───────────┴───────────┐
       │ morph value compose   │   morph_x = knob + CV·0.2 + LFO·depth
       │                       │   morph_y = knob + CV·0.2
       └───────────┬───────────┘
                   │
                   ▼
       ┌─────────────────────────────────┐
       │ MicroInterpreter::Invoke()      │
       │  input = (phase, mx, my)        │   ← 9-anchor SIREN tflite, ~41 KB
       │  output = sample in [-1, +1]    │   ~2-4 µs per call on Apple silicon
       └───────────┬─────────────────────┘
                   │
                   ▼
       out_ = clamp(y, ±1) · 5 V        ← cached for host's get_output()
```

## Where to look in the code

| If you want to … | Read |
|---|---|
| Understand the per-sample DSP | `tf_osc.cc :: update()` |
| Change the knob/jack layout | `tf_osc_elements.cc` |
| Retrain or replace the model | `models/train/train_siren_morph9.py` |
| See how TFLM is linked in | `CMakeLists.txt` (top half) |
| Understand how this gets into the sim | `docs/SIMULATOR_PATCHES.md` |
| Add a new audio test | `tests/test_tfosc.py` + helpers in `conftest.py` |
| Add a new demo WAV | `tests/generate_samples.py` |
| Continue the project with a Claude session | `CLAUDE.md` + the skills under `.claude/skills/` |

## Origins

This started as an experiment to answer: "Can you usefully run a TensorFlow
Lite Micro model at audio rate inside a Eurorack plugin running on an
embedded Cortex-A7?" The answer so far is *yes inside the simulator, with
unknown headroom on real hardware*. The interesting parts of the journey:

- Vendoring TFLite Micro as a self-contained static lib (the upstream repo
  uses Bazel and pulls flatbuffers/ruy/gemmlowp via the build, not git
  submodules — those had to be added by hand).
- Fixing the MetaModule simulator's headless build (ThorVG/FontStash/lvgl
  symbols and ext-plugin registration were missing).
- Discovering that the GUI sim's audio chain requires `mapped_outs` plus a
  `HubMedium` module 0 in the patch YAML — without the Hub the patch loads
  but every panel jack stays muted.
- Replacing a ReLU MLP with a SIREN — `sin(ω·Wx)` activations are
  dramatically better at representing periodic waveforms, with no parameter
  count increase.

For the full history of design moves and dead ends, see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Continuing the project

Run a Claude Code session in this directory and ask for what you want. The
project ships with:

- `CLAUDE.md` — concise project memory pre-loaded into every session.
- `.claude/skills/tfexample-*` — slash-commands for the common workflows
  (train, build, test, render demos, run sim).
- `.claude/agents/tflm-synth-engineer.md` — a domain-aware subagent that
  carries the full architectural context.

The most natural next steps are listed in
[docs/FUTURE_WORK.md](docs/FUTURE_WORK.md). The single most important
unverified question is whether the per-sample TFLM `Invoke()` makes audio
budget on real Cortex-A7 hardware — see the *Hardware validation* section.

## Licensing & attribution

- TFExample plugin code: same license as the parent
  [metamodule-plugin-examples](https://github.com/4ms/metamodule-plugin-examples)
  repo.
- Vendored TFLite Micro: Apache 2.0 (upstream
  [google/tflite-micro](https://github.com/tensorflow/tflite-micro)).
- Vendored flatbuffers / ruy / gemmlowp headers: Apache 2.0.
- SIREN initialization recipe: Sitzmann et al. 2020, *Implicit Neural
  Representations with Periodic Activation Functions*.
