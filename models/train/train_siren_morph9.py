"""SIREN wavetable morph with a 3x3 anchor grid (9 timbres).

Anchor layout on the (morph_x, morph_y) plane:

  y=1.0   triangle      vocal_ee       sub_organ
  y=0.5   vocal_ah      fm_2to3        bell
  y=0.0   sine          saw            pulse_15
          x=0.0         x=0.5          x=1.0

Target during training: bilinear interpolation between the 4 nearest
anchors at the queried (x, y).

Architecture: 3 -> 96 -> 96 -> 1 SIREN, sin activations with omega_0 = 30.
Output: float .tflite using only FullyConnected + Sin ops.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

OUT_DIR = Path(__file__).resolve().parent.parent
OUT_TFLITE = OUT_DIR / "wavetable_morph.tflite"
OMEGA_0 = 30.0


# ---------------------------------------------------------------------------
# Nine anchor waveforms — all return [-1, 1]-normalized for phase in [0, 1).
# ---------------------------------------------------------------------------

def _norm(y: np.ndarray) -> np.ndarray:
    return y / np.max(np.abs(y) + 1e-9)


def w_sine(phase):
    return np.sin(2 * np.pi * phase)


def w_triangle(phase):
    return 2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0


def w_saw(phase, n_harm=16):
    y = np.zeros_like(phase)
    for k in range(1, n_harm + 1):
        y += (1.0 / k) * np.sin(2 * np.pi * k * phase)
    return _norm(y)


def w_pulse(phase, duty=0.15, n_harm=20):
    """Bandlimited rectangular pulse."""
    # DC offset removed; additive synthesis.
    y = np.zeros_like(phase)
    for k in range(1, n_harm + 1):
        y += (2.0 / (np.pi * k)) * np.sin(np.pi * k * duty) * np.cos(2 * np.pi * k * phase)
    return _norm(y)


def _formant_wave(phase, formants, n_harm=24, f0=200.0):
    """Sum harmonics weighted by Lorentzian formant filter."""
    def gain(freq):
        g = 0.0
        for f, bw in formants:
            g += 1.0 / (1.0 + ((freq - f) / bw) ** 2)
        return g
    y = np.zeros_like(phase)
    for k in range(1, n_harm + 1):
        y += gain(k * f0) * np.sin(2 * np.pi * k * phase)
    return _norm(y)


def w_vocal_ah(phase):
    return _formant_wave(phase, [(730, 90), (1090, 110), (2440, 170)])


def w_vocal_ee(phase):
    # F1 lowered, F2 raised — classic "EE" vowel.
    return _formant_wave(phase, [(270, 70), (2290, 130), (3010, 160)])


def w_fm_2to3(phase, mod_idx=2.5):
    """Carrier:modulator = 2:3 FM, fairly clangorous."""
    mod = np.sin(2 * np.pi * 3.0 * phase)
    y = np.sin(2 * np.pi * 2.0 * phase + mod_idx * mod)
    return _norm(y)


def w_bell(phase, n_partials=6):
    ratios = [1.0, 2.756, 5.404, 8.933, 13.345, 18.638][:n_partials]
    y = np.zeros_like(phase)
    for i, r in enumerate(ratios):
        y += (1.0 / (i + 1)) * np.sin(2 * np.pi * r * phase)
    return _norm(y)


def w_sub_organ(phase):
    """Pipe-organ stack: sub + fundamental + 5th + octave (Hammond-ish)."""
    y = ( 1.0  * np.sin(2 * np.pi * 0.5 * phase)   # sub octave
        + 1.0  * np.sin(2 * np.pi * 1.0 * phase)   # fundamental
        + 0.6  * np.sin(2 * np.pi * 1.5 * phase)   # 5th
        + 0.8  * np.sin(2 * np.pi * 2.0 * phase)   # octave
        + 0.4  * np.sin(2 * np.pi * 3.0 * phase))
    return _norm(y)


# Grid layout: rows are y, columns are x. Each cell holds a waveform fn.
GRID = [
    [w_sine,     w_saw,      w_pulse    ],   # y = 0.0
    [w_vocal_ah, w_fm_2to3,  w_bell     ],   # y = 0.5
    [w_triangle, w_vocal_ee, w_sub_organ],   # y = 1.0
]
GRID_X = [0.0, 0.5, 1.0]   # 3 columns
GRID_Y = [0.0, 0.5, 1.0]   # 3 rows


def morph_target(phase, x, y):
    """Bilinear interp over the four nearest grid anchors at (x, y)."""
    # Find cell. With GRID_X = [0, 0.5, 1.0]: x ∈ [0, 0.5) → col 0, x ∈ [0.5, 1] → col 1.
    cx = np.clip(np.searchsorted(GRID_X, x, side="right") - 1, 0, len(GRID_X) - 2)
    cy = np.clip(np.searchsorted(GRID_Y, y, side="right") - 1, 0, len(GRID_Y) - 2)
    x0, x1 = np.take(GRID_X, cx), np.take(GRID_X, cx + 1)
    y0, y1 = np.take(GRID_Y, cy), np.take(GRID_Y, cy + 1)
    fx = (x - x0) / (x1 - x0 + 1e-9)
    fy = (y - y0) / (y1 - y0 + 1e-9)

    out = np.zeros_like(phase)
    for cy_i in range(len(GRID_Y) - 1):
        for cx_i in range(len(GRID_X) - 1):
            mask = (cx == cx_i) & (cy == cy_i)
            if not np.any(mask):
                continue
            ph = phase[mask]
            f00 = GRID[cy_i    ][cx_i    ](ph)
            f10 = GRID[cy_i    ][cx_i + 1](ph)
            f01 = GRID[cy_i + 1][cx_i    ](ph)
            f11 = GRID[cy_i + 1][cx_i + 1](ph)
            wx = fx[mask]; wy = fy[mask]
            out[mask] = ((1 - wx) * (1 - wy) * f00
                         + wx     * (1 - wy) * f10
                         + (1 - wx) * wy     * f01
                         + wx     * wy       * f11)
    return out


# ---------------------------------------------------------------------------
# SIREN model.
# ---------------------------------------------------------------------------

class SirenInit(tf.keras.initializers.Initializer):
    def __init__(self, in_features, is_first, omega_0=OMEGA_0):
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0

    def __call__(self, shape, dtype=None):
        if self.is_first:
            limit = 1.0 / self.in_features
        else:
            limit = np.sqrt(6.0 / self.in_features) / self.omega_0
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)


def build_siren(input_dim=3, hidden=96, n_hidden=2, omega_0=OMEGA_0):
    inp = tf.keras.layers.Input(shape=(input_dim,))
    x = inp
    for i in range(n_hidden):
        in_features = input_dim if i == 0 else hidden
        x = tf.keras.layers.Dense(
            hidden,
            kernel_initializer=SirenInit(in_features, is_first=(i == 0), omega_0=omega_0),
            bias_initializer="zeros",
        )(x)
        x = tf.keras.layers.Lambda(lambda t: tf.math.sin(omega_0 * t))(x)
    out = tf.keras.layers.Dense(
        1,
        kernel_initializer=SirenInit(hidden, is_first=False, omega_0=omega_0),
        bias_initializer="zeros",
    )(x)
    return tf.keras.Model(inp, out)


# ---------------------------------------------------------------------------
# Train.
# ---------------------------------------------------------------------------

N_SAMPLES = 400_000
rng = np.random.default_rng(seed=42)
phase = rng.random(N_SAMPLES).astype(np.float32)
x = rng.random(N_SAMPLES).astype(np.float32)
y = rng.random(N_SAMPLES).astype(np.float32)
target = morph_target(phase, x, y).astype(np.float32)

X = np.stack([phase, x, y], axis=1)
Y = target[:, None]

print(f"Training set: X={X.shape}, Y={Y.shape}, target range=[{Y.min():.3f}, {Y.max():.3f}]")

model = build_siren()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
model.summary()
history = model.fit(X, Y, batch_size=512, epochs=120, validation_split=0.05, verbose=2)
final_loss = history.history["val_loss"][-1]
print(f"Final val_loss: {final_loss:.5f}  →  RMS error ≈ {np.sqrt(final_loss):.4f}")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []
tflite_bytes = converter.convert()
OUT_TFLITE.write_bytes(tflite_bytes)
print(f"\nWrote {OUT_TFLITE} ({len(tflite_bytes)} bytes)")

interp = tf.lite.Interpreter(model_content=tflite_bytes)
interp.allocate_tensors()
in_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
print(f"Input: shape={in_det['shape']}, dtype={in_det['dtype']}")
print(f"Output: shape={out_det['shape']}, dtype={out_det['dtype']}")

ops_used = sorted({op["op_name"] for op in interp._get_ops_details()})
print(f"TFLite ops used: {ops_used}")

# Per-anchor correlation probe.
phases = np.linspace(0, 1, 1024, endpoint=False).astype(np.float32)
anchor_names = [["sine", "saw", "pulse"],
                ["vocal_ah", "fm_2to3", "bell"],
                ["triangle", "vocal_ee", "sub_organ"]]
print("\nPer-anchor correlation (truth vs SIREN prediction):")
for j, ay in enumerate(GRID_Y):
    for i, ax in enumerate(GRID_X):
        truth = morph_target(phases, np.full_like(phases, ax), np.full_like(phases, ay))
        pred = np.empty_like(truth)
        for k, p in enumerate(phases):
            interp.set_tensor(in_det["index"], np.array([[p, ax, ay]], dtype=np.float32))
            interp.invoke()
            pred[k] = interp.get_tensor(out_det["index"])[0, 0]
        corr = np.corrcoef(truth, pred)[0, 1]
        print(f"  ({ax:.1f},{ay:.1f}) {anchor_names[j][i]:11s}  corr={corr:+.4f}  "
              f"truth_pk={np.max(np.abs(truth)):.3f}  pred_pk={np.max(np.abs(pred)):.3f}")
