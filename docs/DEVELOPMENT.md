# TFExample Development Workflow

Everything you need to retrain, rebuild, test, and play with TFOsc.

## One-time setup

### 1. arm-none-eabi-gcc 12.3 (for the hardware plugin build)

Download from <https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads>,
install. On macOS it goes to `/Applications/ArmGNUToolchain/12.3.rel1/arm-none-eabi/bin/`.
Pass it to CMake with `-DTOOLCHAIN_BASE_DIR=...`.

### 2. Host toolchain (for the simulator)

```bash
brew install cmake ninja sdl2
xcode-select --install
```

### 3. Python venv for training + testing

```bash
cd TFExample/models/train
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install tensorflow scipy pytest
```

TF 2.21+ on Python 3.13 works. Earlier Python should also be fine.

### 4. MetaModule simulator (separate repo)

```bash
git clone --depth 1 --recurse-submodules --shallow-submodules \
  https://github.com/4ms/metamodule.git \
  /Users/marco/ia/metamodule/metamodule
```

The headless build needs three local patches; see [SIMULATOR_PATCHES.md](SIMULATOR_PATCHES.md).
Then:

```bash
cd metamodule/simulator
# GUI build (lets you click around):
cmake --fresh --preset Default && cmake --build build
# OR headless build (for tests and demo rendering):
cmake --fresh --preset headless-min && cmake --build build
```

## Common workflows

### Build only the hardware plugin (.mmplugin)

```bash
cd TFExample
cmake --fresh -B build -G Ninja \
  -DTOOLCHAIN_BASE_DIR=/Applications/ArmGNUToolchain/12.3.rel1/arm-none-eabi/bin
cmake --build build
ls ../metamodule-plugins/TFExample.mmplugin
```

The `.mmplugin` is what you copy to a real MetaModule (or the simulator
would normally load it, but the sim uses the static-link path described
below).

### Retrain the SIREN model

```bash
cd TFExample/models/train
.venv/bin/python3 train_siren_morph9.py
# new wavetable_morph.tflite gets written next to it
```

Then re-embed as a C array:

```bash
cd TFExample/models
xxd -i -n g_wavetable_morph_model_data wavetable_morph.tflite > .raw
{
  echo '#include "wavetable_morph_model_data.h"'
  echo
  echo 'alignas(8) const unsigned char g_wavetable_morph_model_data[] = {'
  awk '/^unsigned char/{flag=1; next} /^unsigned int/{flag=0} flag && !/^};/' .raw
  echo '};'
  echo
  awk '/^unsigned int/{print "const unsigned int g_wavetable_morph_model_data_len = " $NF}' .raw
} > wavetable_morph_model_data.cc
rm .raw
```

(Or use the `/tfexample-train` Claude skill.)

After re-embedding, rebuild both targets.

### Run audio tests

```bash
cd TFExample/tests
../models/train/.venv/bin/pytest -v
```

13 tests cover: non-silent output, voltage rails, L/R match, freq-knob
log mapping (5 points), morph plane spectral content, fine-tune semitone
accuracy, LFO modulation, warp harmonic injection, FM-amount inertness
without input.

Requires the simulator built at `metamodule/simulator/build/simulator`.
Set `METAMODULE_SIM_DIR=/path/to/simulator` to override.

### Regenerate demo WAVs

```bash
cd TFExample/tests
../models/train/.venv/bin/python3 generate_samples.py /tmp/tfosc_samples
for f in /tmp/tfosc_samples/*.wav; do
  echo "▸ $(basename $f)"; afplay "$f"
done
```

30 WAVs covering anchors, morph sweeps, tuning, LFO, warp, and composite
musical figures.

### Run the GUI simulator interactively

```bash
cd metamodule/simulator
# Default preset (GUI):
cmake --fresh --preset Default && cmake --build build
./build/simulator --assets build/assets.uimg --zoom 200
```

In the SDL window:

| Key | Action |
|---|---|
| `1` | Lock audio routing to Out 1 → L, Out 2 → R (do this first) |
| `↑` | Back / aux button (navigates up out of patch view) |
| `←` / `→` | Encoder rotate |
| `↓` | Encoder press |
| `a` `b` `c` `d` `e` `f` | Select knob A–F |
| `[` / `]` | Decrease / increase selected knob by 5 % |
| `Escape` | Quit |

To play TFOsc: press `1`, escape to patch selector, open `tfosc_test.yml`,
rotate encoder to highlight the Play button, press encoder. Then `a` / `b` /
`c` / `d` / ... select knobs; `[` / `]` to turn them.

### Run a custom patch headless

```bash
cd metamodule/simulator
./build/simulator -p patches/tfosc_test.yml -n 48000 -o /tmp/out.raw
# /tmp/out.raw is a WAV file (despite the .raw extension) — float32 stereo
```

### Modify the panel layout

Edit `TFExample/tf_osc_elements.cc`. Each element has `x_mm`, `y_mm`,
`short_name`, and `image`. The panel is 10 HP wide (≈50 mm) by ~110 mm
tall. Knobs sit at two columns: `x=15` and `x=35`. After editing rebuild
and re-run the sim — the new panel renders immediately.

### Add a new control (knob or jack)

1. Add an ID to the enum in `tf_osc.hh`.
2. Add a member variable to hold the value.
3. Wire it in `set_param()` or `set_input()` in `tf_osc.cc`.
4. Use it inside `update()`.
5. Add a new `MetaModule::Knob` / `JackInput` entry in
   `tf_osc_elements.cc`. Bump the array sizes at the top.
6. If it's a knob with a default, add an entry to `static_knobs` in any
   patch YAML that loads TFOsc.
7. Add the param name → ID mapping in `tests/conftest.py :: PARAM`.
8. Add a test in `tests/test_tfosc.py`.

### Add a new TFLM operator

The plugin's `MicroMutableOpResolver` template parameter is the kernel
count. To add another op:

1. Bump the template parameter in `tf_osc.hh` (e.g. `<2>` → `<3>`).
2. Call `resolver_.AddNewOp()` in `tf_osc.cc`'s constructor.
3. Ensure the kernel's `.cc` file is compiled in
   `TFExample/CMakeLists.txt` under the `tflite_micro` target. Most
   live in `third_party/tflite-micro/tensorflow/lite/micro/kernels/`.
4. Rebuild.

### Switch between GUI and headless sim builds

Both presets share the same `build/` directory, so switching means a
fresh configure:

```bash
cd metamodule/simulator
cmake --fresh --preset Default        # GUI
# OR
cmake --fresh --preset headless-min   # CI tests / demo rendering
cmake --build build
```

If you forget the `--fresh` you'll get a stale build with the previous
preset's cache.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `cstdint not found`, `-mthumb-interwork unknown` in clangd | clangd using host clang for cross-compile flags | Cosmetic; ignore. Real build is fine. |
| Headless sim writes silent WAV | Patch missing HubMedium at `module_id: 0` | Add `0: '4msCompany:HubMedium'` to `module_slugs` |
| GUI sim plays but no audio | Wrong audio routing (out of band) | Press `1` to reset to Out 1/2 → L/R |
| GUI sim shows patch but Play does nothing | Focus not on Play button | Turn encoder until Play has the highlight border, then press encoder |
| `Module TFExample:TFOsc not found` in headless | `load_ext_builtin_plugins()` not called in main-headless.cc | Apply the `simulator/src/headless/main-headless.cc` patch in SIMULATOR_PATCHES.md |
| Build fails: `FLATBUFFERS_VERSION_MAJOR == 25 && _MINOR == 9` | Vendored flatbuffers headers wrong version | Re-vendor from `git checkout v25.9.23` in flatbuffers |
| `Process time: 0 ms` and silent output | TFOsc not constructed; module became `NullModule` | Same as "Module not found" — check ext-plugin registration |
| Build error: `tensorflow/compiler/mlir/lite/...` not found | Newer upstream tflite-micro layout, slice missing those headers | Copy them from a fresh clone (~10 files under `tensorflow/compiler/mlir/lite/`) |

## CI considerations (not set up)

A future GitHub Action could run, in order:

1. Set up Python + TF + arm-none-eabi-gcc + SDL2.
2. Build the hardware `.mmplugin` (just confirms it links).
3. Clone metamodule, apply the headless-build patches, build `headless-min`.
4. Run `pytest tests/`.

The whole loop takes about 15 minutes locally on Apple silicon; CI
runners would be similar.
