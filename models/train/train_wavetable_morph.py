"""Train a small MLP that maps (phase, morph_x, morph_y) -> sample,
representing a 2D wavetable morph between four anchor waveforms placed
at the corners of the unit square.

Corners:
  (0, 0) sine       — pure tone
  (1, 0) bright saw — rich harmonic stack
  (0, 1) vocal "ah" — formant-shaped harmonic stack
  (1, 1) bell       — inharmonic decay

Training targets are bilinear blends of the four corner waveforms.
Output: float .tflite suitable for TensorFlow Lite Micro with only
FullyConnected ops (ReLU hidden, linear output).
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent
OUT_TFLITE = OUT_DIR / "wavetable_morph.tflite"

# ---------------------------------------------------------------------------
# Waveform recipes — all take phase in [0, 1), return amplitude in roughly [-1, 1].
# ---------------------------------------------------------------------------

def w_sine(phase):
    return np.sin(2 * np.pi * phase)

def w_saw(phase, n_harm=16):
    # Bandlimited additive sawtooth: sum_{k=1..N} (1/k) sin(2π k phase)
    y = np.zeros_like(phase)
    for k in range(1, n_harm + 1):
        y += (1.0 / k) * np.sin(2 * np.pi * k * phase)
    return y / np.max(np.abs(y) + 1e-9)

def w_vocal_ah(phase, n_harm=24, f0=200.0):
    # Three-formant vowel "ah"-like spectrum at f0 = 200 Hz.
    # Formants (Hz, bw): F1=730/90, F2=1090/110, F3=2440/170
    formants = [(730, 90), (1090, 110), (2440, 170)]
    def formant_gain(freq):
        g = 0.0
        for (f, bw) in formants:
            # simple lorentzian
            g += 1.0 / (1.0 + ((freq - f) / bw) ** 2)
        return g
    y = np.zeros_like(phase)
    for k in range(1, n_harm + 1):
        amp = formant_gain(k * f0)
        y += amp * np.sin(2 * np.pi * k * phase)
    return y / np.max(np.abs(y) + 1e-9)

def w_bell(phase, n_partials=6):
    # Inharmonic partials (tubular bell-ish ratios), decaying amplitudes.
    ratios = [1.0, 2.756, 5.404, 8.933, 13.345, 18.638][:n_partials]
    y = np.zeros_like(phase)
    for i, r in enumerate(ratios):
        amp = 1.0 / (i + 1)
        y += amp * np.sin(2 * np.pi * r * phase)
    return y / np.max(np.abs(y) + 1e-9)

CORNER_FNS = [
    # ((x, y), fn)
    ((0.0, 0.0), w_sine),
    ((1.0, 0.0), w_saw),
    ((0.0, 1.0), w_vocal_ah),
    ((1.0, 1.0), w_bell),
]

def morph_target(phase, x, y):
    """Bilinear interp of the four corner waveforms."""
    s = w_sine(phase)
    sw = w_saw(phase)
    vo = w_vocal_ah(phase)
    be = w_bell(phase)
    return ((1 - x) * (1 - y) * s
            + x * (1 - y) * sw
            + (1 - x) * y * vo
            + x * y * be)

# ---------------------------------------------------------------------------
# Training data — random (phase, x, y) triples, target = bilinear blend.
# ---------------------------------------------------------------------------

N_SAMPLES = 200_000
rng = np.random.default_rng(seed=42)
phase = rng.random(N_SAMPLES).astype(np.float32)
x = rng.random(N_SAMPLES).astype(np.float32)
y = rng.random(N_SAMPLES).astype(np.float32)
target = morph_target(phase, x, y).astype(np.float32)

X = np.stack([phase, x, y], axis=1)         # shape (N, 3)
Y = target[:, None]                          # shape (N, 1)

print(f"Training set: X={X.shape}, Y={Y.shape}, target range=[{Y.min():.3f}, {Y.max():.3f}]")

# ---------------------------------------------------------------------------
# Model — 3 -> 64 -> 64 -> 1, ReLU hidden, linear output.
# Only FullyConnected op needed at inference time (TFLM-friendly).
# ---------------------------------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation=None),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
model.summary()

# Train
history = model.fit(X, Y, batch_size=512, epochs=60, validation_split=0.05, verbose=2)
final_loss = history.history["val_loss"][-1]
print(f"Final val_loss (MSE): {final_loss:.5f}  →  RMS error ≈ {np.sqrt(final_loss):.4f}")

# ---------------------------------------------------------------------------
# Convert to float TFLite (no quantization — Cortex-A7 hardfloat is happy).
# ---------------------------------------------------------------------------

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []  # explicitly no quantization
tflite_bytes = converter.convert()

OUT_TFLITE.write_bytes(tflite_bytes)
print(f"\nWrote {OUT_TFLITE} ({len(tflite_bytes)} bytes)")

# Sanity-check round-trip with the TFLite interpreter on a few test points.
interp = tf.lite.Interpreter(model_content=tflite_bytes)
interp.allocate_tensors()
in_det  = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
print(f"Input tensor: shape={in_det['shape']}, dtype={in_det['dtype']}")
print(f"Output tensor: shape={out_det['shape']}, dtype={out_det['dtype']}")

probes = [
    ("sine corner   (0,0)", [0.25, 0.0, 0.0]),
    ("saw corner    (1,0)", [0.25, 1.0, 0.0]),
    ("vocal corner  (0,1)", [0.25, 0.0, 1.0]),
    ("bell corner   (1,1)", [0.25, 1.0, 1.0]),
    ("center        (.5,.5)", [0.25, 0.5, 0.5]),
]
for label, inp in probes:
    interp.set_tensor(in_det["index"], np.array([inp], dtype=np.float32))
    interp.invoke()
    pred = interp.get_tensor(out_det["index"])[0, 0]
    truth = morph_target(np.array([inp[0]]), np.array([inp[1]]), np.array([inp[2]]))[0]
    print(f"  {label}: pred={pred:+.4f}  truth={truth:+.4f}  err={pred-truth:+.4f}")
