"""SIREN variant of the wavetable-morph trainer.

SIREN (Sitzmann et al. 2020) uses periodic sin() activations and a
specific weight initialization that lets a small MLP represent
high-frequency / periodic functions cleanly — the natural fit for
audio waveforms.

Architecture: 3 -> 64 -> 64 -> 1
  - Hidden layers: y = sin(omega_0 * (Wx + b))
  - Output layer:  linear
  - First-layer weights ~ U(-1/in,  1/in)
  - Other hidden    weights ~ U(-sqrt(6/in)/omega_0,  +sqrt(6/in)/omega_0)
  - omega_0 = 30 (paper default for image/audio)

The omega_0 multiply gets emitted as a Mul op in the resulting .tflite,
so the runtime resolver needs FullyConnected + Mul + Sin.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent
OUT_TFLITE = OUT_DIR / "wavetable_morph.tflite"

OMEGA_0 = 30.0

# ---------------------------------------------------------------------------
# Waveform recipes — same as the ReLU MLP version. Vectorized over phase.
# ---------------------------------------------------------------------------

def w_sine(phase):
    return np.sin(2 * np.pi * phase)

def w_saw(phase, n_harm=16):
    y = np.zeros_like(phase)
    for k in range(1, n_harm + 1):
        y += (1.0 / k) * np.sin(2 * np.pi * k * phase)
    return y / np.max(np.abs(y) + 1e-9)

def w_vocal_ah(phase, n_harm=24, f0=200.0):
    formants = [(730, 90), (1090, 110), (2440, 170)]
    def formant_gain(freq):
        g = 0.0
        for f, bw in formants:
            g += 1.0 / (1.0 + ((freq - f) / bw) ** 2)
        return g
    y = np.zeros_like(phase)
    for k in range(1, n_harm + 1):
        y += formant_gain(k * f0) * np.sin(2 * np.pi * k * phase)
    return y / np.max(np.abs(y) + 1e-9)

def w_bell(phase, n_partials=6):
    ratios = [1.0, 2.756, 5.404, 8.933, 13.345, 18.638][:n_partials]
    y = np.zeros_like(phase)
    for i, r in enumerate(ratios):
        y += (1.0 / (i + 1)) * np.sin(2 * np.pi * r * phase)
    return y / np.max(np.abs(y) + 1e-9)

def morph_target(phase, x, y):
    s  = w_sine(phase)
    sw = w_saw(phase)
    vo = w_vocal_ah(phase)
    be = w_bell(phase)
    return ((1 - x) * (1 - y) * s
            + x * (1 - y) * sw
            + (1 - x) * y * vo
            + x * y * be)

# ---------------------------------------------------------------------------
# SIREN layers.
# ---------------------------------------------------------------------------

class SirenInit(tf.keras.initializers.Initializer):
    """Kernel init per the SIREN paper."""
    def __init__(self, in_features: int, is_first: bool, omega_0: float = OMEGA_0):
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0

    def __call__(self, shape, dtype=None):
        if self.is_first:
            limit = 1.0 / self.in_features
        else:
            limit = np.sqrt(6.0 / self.in_features) / self.omega_0
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)


def build_siren(input_dim=3, hidden=64, n_hidden_layers=2, omega_0=OMEGA_0):
    inp = tf.keras.layers.Input(shape=(input_dim,))
    x = inp
    for i in range(n_hidden_layers):
        in_features = input_dim if i == 0 else hidden
        is_first = (i == 0)
        x = tf.keras.layers.Dense(
            hidden,
            kernel_initializer=SirenInit(in_features, is_first, omega_0),
            bias_initializer="zeros",
        )(x)
        x = tf.keras.layers.Lambda(lambda t: tf.math.sin(omega_0 * t))(x)
    # Linear output.
    out = tf.keras.layers.Dense(
        1,
        kernel_initializer=SirenInit(hidden, False, omega_0),
        bias_initializer="zeros",
    )(x)
    return tf.keras.Model(inp, out)


# ---------------------------------------------------------------------------
# Training data.
# ---------------------------------------------------------------------------

N_SAMPLES = 200_000
rng = np.random.default_rng(seed=42)
phase = rng.random(N_SAMPLES).astype(np.float32)
x = rng.random(N_SAMPLES).astype(np.float32)
y = rng.random(N_SAMPLES).astype(np.float32)
target = morph_target(phase, x, y).astype(np.float32)

X = np.stack([phase, x, y], axis=1)
Y = target[:, None]

print(f"Training set: X={X.shape}, Y={Y.shape}, target range=[{Y.min():.3f}, {Y.max():.3f}]")

# ---------------------------------------------------------------------------
# Train.
# ---------------------------------------------------------------------------

model = build_siren()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
model.summary()
history = model.fit(X, Y, batch_size=512, epochs=80, validation_split=0.05, verbose=2)
final_loss = history.history["val_loss"][-1]
print(f"Final val_loss (MSE): {final_loss:.5f}  →  RMS error ≈ {np.sqrt(final_loss):.4f}")

# ---------------------------------------------------------------------------
# Convert to float TFLite (no quantization).
# ---------------------------------------------------------------------------

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []
tflite_bytes = converter.convert()
OUT_TFLITE.write_bytes(tflite_bytes)
print(f"\nWrote {OUT_TFLITE} ({len(tflite_bytes)} bytes)")

# Round-trip + list the op set so we know what kernels the resolver needs.
interp = tf.lite.Interpreter(model_content=tflite_bytes)
interp.allocate_tensors()
in_det  = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
print(f"Input  : shape={in_det['shape']}, dtype={in_det['dtype']}")
print(f"Output : shape={out_det['shape']}, dtype={out_det['dtype']}")

ops_used = sorted({op["op_name"] for op in interp._get_ops_details()})
print(f"TFLite ops used: {ops_used}")

# Quick spectral sanity probe at a few morph corners — feed 1024 phase samples
# and compare to ground truth via correlation.
phases = np.linspace(0, 1, 1024, endpoint=False).astype(np.float32)
for label, (mx, my) in [("sine",  (0.0, 0.0)),
                         ("saw",   (1.0, 0.0)),
                         ("vocal", (0.0, 1.0)),
                         ("bell",  (1.0, 1.0))]:
    truth = morph_target(phases, np.full_like(phases, mx), np.full_like(phases, my))
    pred = np.empty_like(truth)
    for i, p in enumerate(phases):
        interp.set_tensor(in_det["index"], np.array([[p, mx, my]], dtype=np.float32))
        interp.invoke()
        pred[i] = interp.get_tensor(out_det["index"])[0, 0]
    corr = np.corrcoef(truth, pred)[0, 1]
    print(f"  {label:5s} corner: corr(truth, pred) = {corr:+.4f}  "
          f"truth_peak={np.max(np.abs(truth)):.3f}  pred_peak={np.max(np.abs(pred)):.3f}")
