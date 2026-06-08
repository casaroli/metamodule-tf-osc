---
name: tfexample-test
description: Run the pytest audio test suite for TFExample. Each test renders a patch through the headless simulator and asserts on level / pitch / spectral content. Use when the user wants to validate that the plugin still works after a code change.
---

# Run TFExample audio test suite

13 end-to-end audio tests under `tests/test_tfosc.py`. They drive the
**headless** simulator (must be built first) and assert on the WAV output.

## Prerequisites

1. Simulator built with the `headless-min` preset. If not, use the
   `tfexample-build-sim` skill with arg `headless` first.
2. Training venv exists at `models/train/.venv/` (provides pytest +
   numpy + scipy). If not, see `tfexample-train` skill for setup.

## Run

```bash
cd tests
../models/train/.venv/bin/pytest -v
```

All 13 should pass in ~3 seconds. If any fail, examine the FFT/RMS
report in the failure message — it usually tells you whether the audio
chain is broken, the pitch mapping is off, or a control isn't wired.

## Override the simulator location

Set `METAMODULE_SIM_DIR` if the sim isn't at the default
`/Users/marco/ia/metamodule/metamodule/simulator`:

```bash
METAMODULE_SIM_DIR=/path/to/simulator \
  ../models/train/.venv/bin/pytest -v
```

## Test grouping

- `test_audio_*` — baseline (non-silent, voltage rails, L/R match)
- `test_freq_knob_*` — log pitch mapping across 5 knob positions
- `test_morph_*` — spectral content of morph corners
- `test_fine_tune_*` — fine tune knob ≈ ±1 semitone
- `test_lfo_*` — auto-morph LFO modulates spectrum over time
- `test_warp_*` — phase distortion injects harmonics
- `test_fm_amt_*` — FM amount knob is inert without an FM input signal

## When tests are silent (all fail with "RMS=0")

Almost always one of:

1. Plugin not registered with sim's brand factory → check
   `metamodule/simulator/ext-plugins.cmake` includes TFExample.
2. Sim rebuilt with the wrong preset → headless link must include
   the patches per `docs/SIMULATOR_PATCHES.md`.
3. Patch YAML missing HubMedium → see `conftest.py :: tfosc_patch()`,
   it always emits `0: '4msCompany:HubMedium'`. If you customized that,
   restore it.

## When freq tests fail with "got 20.0 Hz"

That's `dominant_freq()`'s noise floor — means the output is all zeros
and the FFT peak is at the search-window lower bound. Same root cause
as "all silent" above.
