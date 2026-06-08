"""Rich demo WAVs for TFOsc — sweeps every controllable surface to give
an audible tour of what the neural wavetable synth can do.

Grouped sections:
  0x   sanity / pure anchors
  1x   morph plane sweeps
  2x   tuning + fine
  3x   auto-morph LFO (built-in)
  4x   phase distortion (warp)
  5x   composite "musical" examples
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import render, tfosc_patch  # noqa: E402

SR = 48000
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/tfosc_samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(path: Path, arr: np.ndarray):
    norm = np.clip(arr / 5.0 * 0.7, -1.0, 1.0)
    wavfile.write(path, SR, (norm * 32767).astype(np.int16))
    print(f"  → {path.name}  ({len(arr)/SR:.2f}s)")


def render_seg(dur_s: float, **knobs) -> np.ndarray:
    return render(tfosc_patch(**knobs), n_samples=int(SR * dur_s))


def sweep(steps, step_dur=0.20):
    return np.concatenate([render_seg(step_dur, **k) for k in steps], axis=0)


def crossfade(a: np.ndarray, b: np.ndarray, ms: int = 30) -> np.ndarray:
    """Linear crossfade between two segments to hide concatenation clicks."""
    n = int(SR * ms / 1000)
    if n >= len(a) or n >= len(b):
        return np.concatenate([a, b])
    ramp = np.linspace(0, 1, n)[:, None]
    a = a.copy(); b = b.copy()
    a[-n:] *= (1 - ramp)
    b[:n] *= ramp
    a[-n:] += b[:n]
    return np.concatenate([a, b[n:]])


# 3x3 anchor grid coordinates: (name, mx, my)
ANCHORS = [
    ("sine",        0.0, 0.0),
    ("saw",         0.5, 0.0),
    ("pulse",       1.0, 0.0),
    ("vocal_ah",    0.0, 0.5),
    ("fm_2to3",     0.5, 0.5),
    ("bell",        1.0, 0.5),
    ("triangle",    0.0, 1.0),
    ("vocal_ee",    0.5, 1.0),
    ("sub_organ",   1.0, 1.0),
]

F = 0.5  # ~200 Hz fundamental for A/B between timbres.

print(f"=== TFOsc rich demos — writing to {OUT_DIR}\n")

# ---------------------------------------------------------------------------
# 0x — Sanity / pure anchors at 200 Hz.
# ---------------------------------------------------------------------------
print("[0x] 9 anchor timbres @ 200 Hz")
for i, (name, mx, my) in enumerate(ANCHORS, start=1):
    save(OUT_DIR / f"0{i}_anchor_{name}.wav",
         render_seg(2.0, freq=F, morph_x=mx, morph_y=my))

# ---------------------------------------------------------------------------
# 1x — Morph plane sweeps.
# ---------------------------------------------------------------------------
print("\n[1x] Morph plane sweeps (4-5 s each)")
xs = np.linspace(0.0, 1.0, 24)
save(OUT_DIR / "10_sweep_bottom_edge.wav",
     sweep([{"freq": F, "morph_x": x, "morph_y": 0.0} for x in xs]))
save(OUT_DIR / "11_sweep_mid_edge.wav",
     sweep([{"freq": F, "morph_x": x, "morph_y": 0.5} for x in xs]))
save(OUT_DIR / "12_sweep_top_edge.wav",
     sweep([{"freq": F, "morph_x": x, "morph_y": 1.0} for x in xs]))
ts = np.linspace(0.0, 1.0, 30)
save(OUT_DIR / "13_diagonal.wav",
     sweep([{"freq": F, "morph_x": t, "morph_y": t} for t in ts]))
save(OUT_DIR / "14_anti_diagonal.wav",
     sweep([{"freq": F, "morph_x": t, "morph_y": 1.0 - t} for t in ts]))
# Spiral around the plane.
print("    spiral path")
angles = np.linspace(0, 4 * np.pi, 40)
spiral = [{"freq": F,
           "morph_x": 0.5 + 0.45 * np.cos(a) * (i / len(angles)),
           "morph_y": 0.5 + 0.45 * np.sin(a) * (i / len(angles))}
          for i, a in enumerate(angles)]
save(OUT_DIR / "15_spiral.wav", sweep(spiral, step_dur=0.15))

# ---------------------------------------------------------------------------
# 2x — Tuning / fine.
# ---------------------------------------------------------------------------
print("\n[2x] Tuning + fine")
# Pitch sweep at FM center
save(OUT_DIR / "20_pitch_sweep_fm.wav",
     sweep([{"freq": f, "morph_x": 0.5, "morph_y": 0.5}
            for f in np.linspace(0.2, 0.85, 28)], step_dur=0.15))
# Fine tune slow sweep — should sound like slow detuning across A
save(OUT_DIR / "21_fine_tune_sweep.wav",
     sweep([{"freq": 0.5, "fine": f, "morph_x": 0.5, "morph_y": 0.0}
            for f in np.linspace(0.0, 1.0, 30)], step_dur=0.15))

# ---------------------------------------------------------------------------
# 3x — Built-in LFO auto-morph at various rates / depths.
# ---------------------------------------------------------------------------
print("\n[3x] Auto-morph LFO")
save(OUT_DIR / "30_lfo_slow_shallow.wav",
     render_seg(6.0, freq=F, morph_x=0.5, morph_y=0.5,
                lfo_rate=0.25, lfo_depth=0.3))
save(OUT_DIR / "31_lfo_slow_deep.wav",
     render_seg(6.0, freq=F, morph_x=0.5, morph_y=0.5,
                lfo_rate=0.25, lfo_depth=0.9))
save(OUT_DIR / "32_lfo_medium.wav",
     render_seg(6.0, freq=F, morph_x=0.3, morph_y=0.7,
                lfo_rate=0.55, lfo_depth=0.7))
save(OUT_DIR / "33_lfo_fast.wav",
     render_seg(4.0, freq=F, morph_x=0.5, morph_y=0.3,
                lfo_rate=0.9, lfo_depth=0.6))
# Three-rate LFO sweep — slow→fast over 6 s.
save(OUT_DIR / "34_lfo_rate_sweep.wav",
     sweep([{"freq": F, "morph_x": 0.5, "morph_y": 0.5,
             "lfo_rate": r, "lfo_depth": 0.7}
            for r in np.linspace(0.0, 1.0, 30)], step_dur=0.20))

# ---------------------------------------------------------------------------
# 4x — Phase distortion (Warp).
# ---------------------------------------------------------------------------
print("\n[4x] Phase distortion (Warp)")
# Hold morph at sine corner and sweep warp → reveals the PD harmonics
save(OUT_DIR / "40_warp_on_sine.wav",
     sweep([{"freq": F, "morph_x": 0.0, "morph_y": 0.0, "warp": w}
            for w in np.linspace(0.5, 0.05, 25)], step_dur=0.16))
save(OUT_DIR / "41_warp_on_saw.wav",
     sweep([{"freq": F, "morph_x": 0.5, "morph_y": 0.0, "warp": w}
            for w in np.linspace(0.5, 0.95, 25)], step_dur=0.16))
# PWM-style: rhythmic warp wobble
save(OUT_DIR / "42_warp_pwm_wobble.wav",
     sweep([{"freq": F, "morph_x": 1.0, "morph_y": 0.0,
             "warp": 0.5 + 0.4 * np.sin(2 * np.pi * t / 30)}
            for t in range(60)], step_dur=0.10))

# ---------------------------------------------------------------------------
# 5x — Composite musical examples.
# ---------------------------------------------------------------------------
print("\n[5x] Composite musical figures")

# Pentatonic notes at various morph + LFO settings
def pentatonic(base_freq=0.5, mx=0.5, my=0.5, lfo_rate=0.0, lfo_depth=0.0,
               warp=0.5, note_dur=0.4):
    semitones = [0, 3, 5, 7, 10, 12, 10, 7, 5, 3]
    notes = [{"freq": base_freq + s / 12.0 / 2.0,
              "morph_x": mx, "morph_y": my,
              "lfo_rate": lfo_rate, "lfo_depth": lfo_depth, "warp": warp}
             for s in semitones]
    # Crossfade between notes for legato.
    parts = [render_seg(note_dur, **k) for k in notes]
    out = parts[0]
    for p in parts[1:]:
        out = crossfade(out, p, ms=20)
    return out

save(OUT_DIR / "50_pentatonic_vocal.wav", pentatonic(mx=0.5, my=1.0))
save(OUT_DIR / "51_pentatonic_bell.wav",  pentatonic(mx=1.0, my=0.5))
save(OUT_DIR / "52_pentatonic_organ_lfo.wav",
     pentatonic(mx=0.7, my=1.0, lfo_rate=0.5, lfo_depth=0.4))

# Evolving pad: hold one note, slow LFO + light warp animation
print("    evolving pad")
save(OUT_DIR / "53_evolving_pad.wav",
     sweep([{"freq": 0.45, "fine": 0.5, "morph_x": 0.4, "morph_y": 0.6,
             "lfo_rate": 0.15, "lfo_depth": 0.5,
             "warp": 0.5 + 0.15 * np.sin(2 * np.pi * t / 40)}
            for t in range(80)], step_dur=0.10))

# "Neural lead" — bright morph + warp swells + LFO shimmer
print("    neural lead")
save(OUT_DIR / "54_neural_lead.wav",
     sweep([{"freq": 0.6 + 0.05 * np.sin(2 * np.pi * t / 35),
             "morph_x": 0.7,
             "morph_y": 0.2 + 0.3 * np.sin(2 * np.pi * t / 25),
             "lfo_rate": 0.6, "lfo_depth": 0.3,
             "warp": 0.5 + 0.2 * np.sin(2 * np.pi * t / 18)}
            for t in range(60)], step_dur=0.10))

# Bell hit: short note at bell corner with very slow LFO
save(OUT_DIR / "55_bell_hit.wav",
     render_seg(3.0, freq=0.45, morph_x=1.0, morph_y=0.5,
                lfo_rate=0.1, lfo_depth=0.2))

print(f"\nDone. Listen on macOS:")
print(f"  for f in {OUT_DIR}/*.wav; do echo \"▸ $(basename $f)\"; afplay $f; done")
