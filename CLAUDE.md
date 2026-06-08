# TFExample — Claude Session Memory

You are working inside the TFExample plugin, a MetaModule plugin that
runs **TensorFlow Lite Micro** inference per audio sample to drive a
neural wavetable-morph oscillator (TFOsc). Read README.md for the
overview; this file is the working context.

## Status — read this first

This is an **early experiment**, NOT a finished plugin. It works
end-to-end in the MetaModule simulator (GUI + headless on host) but has
**never been loaded onto real MetaModule hardware**. Whether per-sample
TFLite Micro inference fits the Cortex-A7 audio budget at 48 kHz is
**unanswered** — the highest-priority next step is hardware validation.

There is plenty of obvious work left: hardware validation, richer
models, int8 quantization, training on recorded waveforms, polyphony,
upstreaming simulator patches, faceplate art, CI. See
`docs/FUTURE_WORK.md`. Frame any deliverable as building on a baseline,
not shipping a product.

## Where everything lives

- **This plugin**: `./`
- **MetaModule plugin SDK**: `../metamodule-plugin-sdk/`
- **MetaModule firmware + simulator (cloned for sim use)**:
  `/Users/marco/ia/metamodule/metamodule/`
- **Training venv**: `models/train/.venv/` (TensorFlow 2.21, Python 3.13)

## Critical file pointers

| Job | File |
|---|---|
| Per-sample DSP + TFLM Invoke | `tf_osc.cc` |
| Knob/jack enums, arena size, member layout | `tf_osc.hh` |
| Panel UI element registration | `tf_osc_elements.cc` |
| Init entry points (hardware + sim) | `plugin.cc` |
| TFLM static lib build + plugin link | `CMakeLists.txt` |
| Current trained model | `models/wavetable_morph.tflite` |
| Embedded model C array | `models/wavetable_morph_model_data.{cc,h}` |
| Trainer | `models/train/train_siren_morph9.py` |
| Audio test suite | `tests/test_tfosc.py` (+ `conftest.py`) |
| Demo WAV generator | `tests/generate_samples.py` |
| Vendored TFLM slice | `third_party/tflite-micro/` |

## Conventions used throughout

- **Param IDs** match `tests/conftest.py :: PARAM`:
  `freq=0, fine=1, morph_x=2, morph_y=3, lfo_rate=4, lfo_depth=5, warp=6, fm_amt=7`.
  Inserting a new knob shifts later IDs — update the patch YAML defaults
  AND the conftest PARAM map.
- **Test patches always include `'4msCompany:HubMedium'` at `module_id: 0`**.
  Without it `mapped_outs` silently produces 0 V on every panel jack.
- **Headless sim writes WAV format despite the `.raw` extension** — parse
  with `data[data.find(b'data')+8:]` as float32 stereo.
- **Sim `host_fileio` resolves paths relative to CWD.** Pass relative
  paths to `-p` and `-o`, with `cwd=metamodule/simulator`.
- **All knob defaults map to [0, 1] in `set_param`**. Internal scaling
  happens in `update()` (e.g. `freq_hz = 20 * 100^knob`).
- **TFLM resolver template arg is the kernel count.** Currently `<2>`
  for FullyConnected + Sin. Bump when adding ops.

## Common workflows (don't reinvent, use the skills)

These all have Claude skills under `.claude/skills/`. The skills are
plain markdown — read them if you need to know exactly what gets run.

- `/tfexample-train` — retrain the SIREN model + re-embed C array
- `/tfexample-build-sim` — build either preset of the simulator
- `/tfexample-test` — run pytest against the headless sim
- `/tfexample-demos` — regenerate the 30-WAV demo pack
- `/tfexample-run-sim` — launch the GUI simulator in background

## Quick-reference gotchas

1. **Diagnostics about `-mthumb-interwork`, `cstdint not found`, missing
   `powf`/`fabsf`/`sinf`** — pre-existing clangd noise from the ARM
   cross-compile flags. Ignore. The actual gcc build is fine.
2. **GUI sim audio routing defaults change with `1`–`8` keys.** If audio
   doesn't play, press `1` to reset to Out 1 → L, Out 2 → R. If still
   silent, the patch needs `mapped_outs` AND HubMedium.
3. **GUI sim's `audio_load = 58` is hardcoded.** Don't trust the % gauge.
4. **Two init symbols** in `plugin.cc`: `extern "C" void init()` for
   hardware, plain `void init_TFExample(rack::plugin::Plugin*)` for sim.
   Don't add `extern "C"` to the sim wrapper (linker expects C++ mangling).
5. **`metamodule/` subrepo is a clone, not a submodule.** Our local
   modifications to it are documented in `docs/SIMULATOR_PATCHES.md`.
   Don't blindly `git pull` over them.
6. **Embedded `wavetable_morph_model_data.cc` is ~125 KB.** When you see
   it in a diff after retraining, the size jump is normal.
7. **Don't commit `build/`, `.venv/`, `metamodule/`, or `/tmp/tfosc_samples/`** —
   see `.gitignore`.

## When the user asks for X, you typically need to…

- **"add a knob/jack"** → 8-step recipe in `docs/DEVELOPMENT.md` →
  "Add a new control"
- **"retrain"** → call `/tfexample-train` skill
- **"test it"** → call `/tfexample-test`
- **"hear samples"** → call `/tfexample-demos` then point user at
  `/tmp/tfosc_samples/`
- **"run the simulator"** → call `/tfexample-run-sim`
- **"add a TFLM op"** → see `docs/DEVELOPMENT.md` → "Add a new TFLM operator"
- **"validate on hardware"** → see `docs/FUTURE_WORK.md` → "Hardware validation"
  (you can't do this from here; user needs the physical device)
- **"upstream the sim changes"** → see `docs/SIMULATOR_PATCHES.md` for
  the diff. The user opens the PR.

## Domain background you should already know

- **MetaModule SDK**: native plugins implement `MetaModule::CoreProcessor`
  with virtual methods `update()` (per-sample), `set_param/input/samplerate`,
  `get_output/led_brightness`. Hardware loader looks up an `init` symbol
  in the `.so` (via `--require-defined=init`).
- **TFLite Micro**: header-only-ish C++ static library. Construct
  `MicroMutableOpResolver<N>` with N ops, then `MicroInterpreter` with
  the model + arena. `AllocateTensors()` once, `Invoke()` per inference.
- **SIREN**: Sitzmann et al. 2020 — MLP with `sin(ω·Wx+b)` activations
  and specific weight init that lets it represent periodic functions
  cleanly. We use ω=30, scaled-uniform init.
- **Casio PD warp**: piecewise-linear phase remap around a controllable
  midpoint. Identity at midpoint=0.5; squeezes phase asymmetrically
  otherwise.

## A note on tone

The user prefers Rasta-flavored prose (see workspace memory) but
**source code, comments, and docs stay in standard technical English**.
The README and docs in this folder are NOT in patois — they're meant for
anyone, including the user during code review and future contributors.
