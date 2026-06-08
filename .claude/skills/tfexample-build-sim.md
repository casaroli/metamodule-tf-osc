---
name: tfexample-build-sim
description: Build the MetaModule simulator with TFExample statically linked in. Use when the user changes plugin C++ code, the model, or wants to switch between GUI and headless builds. Pass argument "gui" (default) or "headless" or "hw" to choose target.
---

# Build the MetaModule simulator

The simulator lives at `/Users/marco/ia/metamodule/metamodule/simulator/`
and statically links TFExample via `ext-plugins.cmake`. Builds always go
to `build/` and require a `--fresh` reconfigure when switching presets.

## Usage

The skill accepts one argument:

- `gui` (default) — Default preset. Full GUI with SDL2, lvgl, ThorVG.
  Launches an interactive window. ~1500 source files, ~3 min cold build.
- `headless` — `headless-min` preset. No GUI, pytest-friendly. ~600 source
  files, ~1.5 min cold build.
- `hw` — Build the hardware `.mmplugin` (no simulator). Cross-compiles
  with arm-none-eabi-gcc to ELF32 little-arm. ~30 s.

## Procedure (gui or headless)

```bash
# Kill any running sim process first
pkill -9 -f "build/simulator" 2>/dev/null; sleep 1

cd /Users/marco/ia/metamodule/metamodule/simulator

# Fresh configure for the chosen preset
cmake --fresh --preset Default       # for gui
# OR
cmake --fresh --preset headless-min  # for headless

cmake --build build
```

Build success: last line should be `[N/N] Linking CXX executable simulator`
plus a benign duplicate-libs warning.

## Procedure (hw)

```bash
cd .
cmake --fresh -B build -G Ninja \
  -DTOOLCHAIN_BASE_DIR=/Applications/ArmGNUToolchain/12.3.rel1/arm-none-eabi/bin
cmake --build build
```

Build success: `Creating plugin at .../metamodule-plugins/TFExample.mmplugin`
plus "All symbols found!" from `check_syms.py`.

## After building

- For **gui**: call the `tfexample-run-sim` skill to launch it
- For **headless**: call the `tfexample-test` skill to run pytest, or
  `tfexample-demos` to regenerate audio samples
- For **hw**: the `.mmplugin` is ready to copy to a real MetaModule

## Common build errors

- **"Module TFExample:TFOsc not found"** at runtime → the plugin built
  but the sim never registered the brand. Verify `ext-plugins.cmake`
  has the three `list(APPEND ext_builtin_brand_*)` lines for TFExample.
- **Undefined ThorVG/FontStash symbols** → simulator patches not applied.
  See `docs/SIMULATOR_PATCHES.md`.
- **`FLATBUFFERS_VERSION_MAJOR == 25 && _MINOR == 9` static_assert** →
  vendored flatbuffers headers are wrong version. Re-vendor from
  `git checkout v25.9.23` in flatbuffers.
- **"no member named 'plugin' in 'Jack'"** in main-headless.cc →
  include order. `plugin/Plugin.hpp` MUST come before
  `ext_plugin_builtin.hh`.
