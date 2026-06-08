# TFExample Architecture

## Goal

Run a TensorFlow Lite Micro inference per audio sample (or per N samples
with interpolation) inside a 4ms MetaModule plugin. The model is the audio
generator; classical DSP is only the thin wrapper around `Invoke()`.

## Stack

```
                    ┌─────────────────────────────────┐
                    │     MetaModule audio engine     │
                    │     (calls update() per sample) │
                    └────────────────┬────────────────┘
                                     │ get_output(jack_id)
                                     │ set_input(jack_id, V)
                                     │ set_param(param_id, normalized)
                                     ▼
┌───────────────────────────────────────────────────────────────┐
│                   TFOsc : CoreProcessor                       │
│                                                               │
│  pre-NN DSP:  pitch → phase → (FM + Warp) → mangled phase     │
│  CV summing:  morph_x_knob + morph_x_cv·0.2 + LFO·depth       │
│  NN:          tflite::MicroInterpreter::Invoke()              │
│  post-NN:     clamp(y, ±1) · 5 V                              │
│                                                               │
│  Tensors:     input  [1, 3]  float                            │
│               output [1, 1]  float                            │
│  Arena:       16 KB (header-allocated, in module instance)    │
│  Resolver:    AddFullyConnected() + AddSin()                  │
└───────────────────────────────────────────────────────────────┘
                                     │
              compile-time link of   │   <CoreModules/CoreProcessor.hh>
                                     ▼
┌───────────────────────────────────────────────────────────────┐
│              MetaModule plugin SDK (cross-compiled)            │
│              arm-none-eabi-gcc 12.3 → ELF32-littlearm .so      │
│              produces TFExample.mmplugin (tar) on disk         │
└───────────────────────────────────────────────────────────────┘
```

## Why these design choices

### Why SIREN instead of ReLU MLP

A vanilla ReLU MLP has a hard time representing periodic functions cleanly
— it has to learn to flatten and reactivate piecewise-linear segments to
approximate a sine curve. We measured this with a 4-corner trainer
(`models/train/train_wavetable_morph.py`): the model produced audible
sine output but morph corners blurred together and the harmonic spectrum
was mush.

SIREN (Sitzmann et al. 2020) uses `sin(ω · Wx + b)` activations with a
specific weight init (`U(-1/in, 1/in)` for the first layer, `U(±√(6/in)/ω)`
elsewhere). With `ω = 30` and the same parameter count, the per-anchor
correlation against ground truth jumps from ~0.5–0.8 to **0.99+**. The
audible difference is dramatic — clean corners, smooth interpolation in
between.

The price is one extra TFLite Micro op (`Sin`). Already vendored in
`elementwise.cc`. No new dependencies.

### Why a vendored TFLite Micro slice, not a submodule

`google/tflite-micro` upstream uses Bazel and pulls flatbuffers, ruy, and
gemmlowp at build time via WORKSPACE — none of them live in the upstream
git tree as submodules. A `git submodule add` of tflite-micro would give
us a directory full of placeholder `BUILD` files where the headers should
be.

So the slice is **manually vendored**:

- `tensorflow/lite/{c,core,kernels,micro,schema}/...` — copied from the
  tflite-micro git tree.
- `tensorflow/compiler/mlir/lite/...` — needed by the newer upstream
  layout (`tflite_types.h`, `error_reporter.h` moved here).
- `third_party/flatbuffers/include/` — flatbuffers headers, pinned to
  v25.9.23 to match the embedded `schema_generated.h` `static_assert`.
- `third_party/gemmlowp/{fixedpoint,internal}/` — required by
  `kernels/internal/common.h`.
- `third_party/ruy/ruy/profiler/instrumentation.h` — referenced by
  `kernels/internal/reference/fully_connected.h`, but only as a header
  decoration (no-op without `RUY_PROFILER` define).
- `signal/micro/kernels/{rfft,irfft}.h` — included transitively by
  `micro_mutable_op_resolver.h` for the `AddRfft()` / `AddIrfft()`
  registrations; we never call them, so the linker dead-strips.

Total: ~5.3 MB of source, hermetic build, no `git submodule init` ever
needed.

### Why HubMedium in every patch

The MetaModule patch player has a hardcoded special-case at
`firmware/src/patch_play/patch_player.hh:168`:

```cpp
// First module is the hub
modules[0] = ModuleFactory::create(PanelDef::typeID);
```

So `module_slugs[0]` is **ignored** — the player always instantiates the
panel/hub itself at index 0. `mapped_outs` entries route panel jack indices
to actual module outputs (e.g. `panel_jack_id: 0 → module_id: 1, jack_id: 0`
sends our TFOsc output to panel jack 1). Without that mapping, the panel
jack stays muted.

Our test patches put `'4msCompany:HubMedium'` at index 0 as a no-op
placeholder and TFOsc at index 1.

### Why two init symbols (`init` and `init_TFExample`)

Hardware loader expects an `extern "C" init` symbol (the SDK's linker
script has `-Wl,--require-defined=init`). The MetaModule simulator's
`ext-plugins.cmake` codegens a different entry point per plugin so they
don't collide when statically linked together:

```cpp
extern void init_TFExample(rack::plugin::Plugin *);
init_TFExample(&internal_plugins.emplace_back("TFExample"));
```

Our `plugin.cc` exposes both: `extern "C" void init()` for hardware, plus
a C++-linkage `void init_TFExample(rack::plugin::Plugin*)` for the sim.
Both forward to `init_tf_osc()`. The hardware linker's `--gc-sections`
strips `init_TFExample` since nothing references it; the sim's static
link keeps it.

### Why an internal LFO (and not just a CV jack)

The LFO knob pair is a **demo affordance**, not a synthesis necessity.
With just CV jacks you'd need to patch in an external LFO module to hear
the morph plane move. The built-in LFO lets a single-module patch
demonstrate the neural plane in motion, which is what makes the audio
demos compelling.

The Morph X / Morph Y CV jacks are still there for "real" patching where
external modulation belongs.

### Why Casio-PD-style warp (not a generic waveshaper)

We want a phase-domain transformation, applied *before* the model sees the
phase. Two reasons:

1. The model is smooth across `phase ∈ [0, 1)`, so re-mapping phase to
   itself nonlinearly stays inside the trained domain.
2. A piecewise-linear warp around a movable midpoint produces the
   familiar PWM/symmetry-modulation character that pairs well with
   morph automation.

The implementation matches Casio's CZ-series phase distortion:

```
phase' = ph * 0.5/warp         if ph < warp
       = 0.5 + (ph-warp)*0.5/(1-warp)   otherwise
```

`warp=0.5` is identity; deviating squeezes either half of the phase
trajectory, which produces brighter harmonics without ever feeding the NN
out-of-distribution inputs.

## Code map

| File | Purpose |
|---|---|
| `tf_osc.hh` | Class declaration, knob/jack ID enums, member layout including the tensor arena |
| `tf_osc.cc` | Constructor (interpreter setup) + per-sample `update()` with all DSP |
| `tf_osc_elements.cc` | UI element registration — knob positions, jack positions, `register_module<TFOsc>` |
| `plugin.cc` | Two init entry points (hardware + sim) |
| `plugin.json` | MetaModule plugin manifest (slug, brand, modules) |
| `plugin-mm.json` | 4ms-specific maintainer metadata |
| `CMakeLists.txt` | Builds `tflite_micro` static lib + `TFExample` plugin lib + invokes `create_plugin()` |
| `models/wavetable_morph.tflite` | Current trained SIREN |
| `models/wavetable_morph_model_data.{cc,h}` | The above embedded as `alignas(8) const unsigned char[]` |
| `models/train/train_siren_morph9.py` | Current trainer (3×3 anchor grid, 96-wide SIREN) |
| `tests/conftest.py` | `render()` and `tfosc_patch()` helpers used by all tests + demos |
| `tests/test_tfosc.py` | 13 audio assertions (level, pitch, spectral) |
| `tests/generate_samples.py` | 30-WAV demo pack generator |
| `third_party/tflite-micro/` | Hand-picked TFLM source slice (no submodule) |

## Memory & timing budget

Measured on Apple silicon (M-class) via the headless simulator:

- **Per-sample inference**: ~2–4 µs (well under the 20.8 µs/sample budget at 48 kHz)
- **Per-buffer load**: ~1.2% (single core, 48 000 samples in 12 ms)
- **`.so` text section**: ~190 KB after `--gc-sections` (4-anchor) / ~210 KB (9-anchor SIREN)
- **`.bss`**: under 1 KB (tensor arena is in the module instance struct, not `.bss`)
- **Tensor arena**: 16 KB allocated (instance member), ~768 B of activations actually used by the 3 → 96 → 96 → 1 SIREN
- **Embedded model**: 41 KB (9-anchor SIREN) in `.rodata`

**Untested on Cortex-A7 hardware.** Cortex-A7 at 800 MHz NEON-VFPv4 has
~3 GFLOPS peak; the SIREN needs ~10 K MACs/inference; at 48 kHz that's
480 MMACs/s = 16 % of peak in the optimistic case. Realistic load with
overhead is likely 25–60 %. If too high, `kInferenceStride` can be raised
to 4/8/16 with linear interpolation between calls (already implemented as
a compile-time constant in `tf_osc.hh`).

## Things that surprised us during the build

These are bear-traps documented for whoever extends this next:

1. **Flatbuffers version pin** — `schema_generated.h` carries a runtime
   `static_assert` against the flatbuffers headers' version macros. Use
   v25.9.23 unless you regenerate the schema.
2. **The MetaModule sim has no `init_TFExample` codegen unless you
   register your plugin in `metamodule/simulator/ext-plugins.cmake`.**
   See `docs/SIMULATOR_PATCHES.md` for the exact change.
3. **The headless sim never auto-registers ext-builtin plugins.** The
   GUI path constructs an `InternalPluginManager` that calls
   `load_ext_builtin_plugins()`; the headless path doesn't. We added that
   call directly in `main-headless.cc`.
4. **`Jack` (in patch-serial namespace `rack::`) shadows `rack::plugin`
   namespace lookup in `main-headless.cc`.** Include `plugin/Plugin.hpp`
   *before* `ext_plugin_builtin.hh`.
5. **`host_fileio` in the headless sim resolves paths relative to CWD.**
   When invoking from pytest, pass relative paths and set `cwd=SIM_DIR`.
6. **The GUI sim shows a hardcoded `audio_load = 58` percent** regardless
   of actual CPU. Don't trust the % display in the patch view as a
   load indicator.
7. **All output jacks default to "unplugged" in the GUI sim.** Only the
   two routed to the soundcard (via the `1`–`8` keys) get marked plugged.
   Output to other jacks shows as 0.
8. **The MetaModule simulator's headless preset has a broken link step
   upstream** — missing ThorVG/FontStash symbols. We patched it; see
   `docs/SIMULATOR_PATCHES.md`.
