"""Shared fixtures for TFOsc audio tests."""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

SIM_DIR = Path(
    os.environ.get(
        "METAMODULE_SIM_DIR",
        "/Users/marco/ia/metamodule/metamodule/simulator",
    )
).resolve()

SIM_BIN = SIM_DIR / "build" / "simulator"
PATCHES_DIR = SIM_DIR / "patches"

# Keep in lockstep with the enum in tf_osc.hh.
PARAM = {
    "freq":      0,
    "fine":      1,
    "morph_x":   2,
    "morph_y":   3,
    "lfo_rate":  4,
    "lfo_depth": 5,
    "warp":      6,
    "fm_amt":    7,
}


def render(patch_yaml: str, n_samples: int = 48000) -> np.ndarray:
    """Drop a patch YAML into the sim, render N samples, return (N, 2) float32."""
    if not SIM_BIN.exists():
        pytest.skip(f"simulator not built at {SIM_BIN}")

    patch_path = PATCHES_DIR / "_pytest_render.yml"
    patch_path.write_text(patch_yaml)
    out_rel = "build/_pytest_out.raw"
    out_abs = SIM_DIR / out_rel
    try:
        result = subprocess.run(
            [str(SIM_BIN), "-p", "patches/_pytest_render.yml",
             "-n", str(n_samples), "-o", out_rel],
            cwd=str(SIM_DIR), check=True, capture_output=True, timeout=30,
        )
        stdout = result.stdout.decode()
        if "Error" in stdout or "Failed" in stdout:
            pytest.fail(f"simulator reported error:\n{stdout}")
        raw = out_abs.read_bytes()
    finally:
        patch_path.unlink(missing_ok=True)
        out_abs.unlink(missing_ok=True)

    data_idx = raw.find(b"data")
    if data_idx < 0:
        pytest.fail("Sim output missing WAV 'data' chunk")
    payload_start = data_idx + 8
    arr = np.frombuffer(raw[payload_start:], dtype=np.float32).reshape(-1, 2)
    assert len(arr) == n_samples, f"expected {n_samples} frames, got {len(arr)}"
    return arr


def dominant_freq(samples: np.ndarray, sr: float = 48000.0) -> float:
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / sr)
    floor = np.searchsorted(freqs, 20.0)
    peak_idx = floor + int(np.argmax(spectrum[floor:]))
    return float(freqs[peak_idx])


def hf_energy(samples: np.ndarray, above_hz: float = 600.0, sr: float = 48000.0) -> float:
    spec = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / sr)
    return float(np.sum(spec[freqs > above_hz] ** 2))


def tfosc_patch(**knobs) -> str:
    """Build a TFOsc-only patch YAML. Pass any subset of:
        freq, fine, morph_x, morph_y, lfo_rate, lfo_depth, warp, fm_amt.
    Unspecified knobs use sensible defaults (no LFO, no warp, no FM).
    """
    defaults = {
        "freq":      0.5,
        "fine":      0.5,   # = 0 detune
        "morph_x":   0.3,
        "morph_y":   0.3,
        "lfo_rate":  0.0,
        "lfo_depth": 0.0,
        "warp":      0.5,   # = identity (PD off)
        "fm_amt":    0.0,
    }
    defaults.update(knobs)

    static_knob_block = "\n".join(
        f"    - module_id: 1\n      param_id: {PARAM[name]}\n      value: {val}"
        for name, val in defaults.items()
    )

    return f"""PatchData:
  patch_name: TFOsc_pytest
  description: pytest TFOsc audio render
  module_slugs:
    0: '4msCompany:HubMedium'
    1: 'TFExample:TFOsc'
  int_cables: []
  mapped_ins: []
  mapped_outs:
    - panel_jack_id: 0
      out:
        module_id: 1
        jack_id: 0
    - panel_jack_id: 1
      out:
        module_id: 1
        jack_id: 0
  static_knobs:
{static_knob_block}
  mapped_knobs: []
  midi_maps:
    name: ''
    set: []
  midi_poly_num: 0
  midi_poly_mode: 0
  midi_pitchwheel_range: 1
  mapped_lights: []
  vcvModuleStates: []
"""
