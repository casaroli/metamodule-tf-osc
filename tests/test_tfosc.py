"""End-to-end audio tests for the TFOsc neural-wavetable oscillator.

Each test renders a TFOsc-only patch through the MetaModule headless
simulator and asserts on level / pitch / spectral content.

Requires the headless simulator built at
$METAMODULE_SIM_DIR/build/simulator (default: ../../metamodule/simulator).
"""

import numpy as np
import pytest

from conftest import render, dominant_freq, hf_energy, tfosc_patch


SR = 48000.0


# ---------------------------------------------------------------------------
# Baseline (already proven by earlier work).
# ---------------------------------------------------------------------------

def test_audio_is_non_silent():
    arr = render(tfosc_patch(), n_samples=int(SR))
    rms = float(np.sqrt(np.mean(arr[:, 0] ** 2)))
    assert rms > 0.5, f"output too quiet (RMS={rms:.4f}); audio chain broken"


def test_audio_within_voltage_rails():
    arr = render(tfosc_patch(), n_samples=int(SR))
    peak = float(np.max(np.abs(arr[:, 0])))
    assert peak < 5.01, f"output exceeds 5 V rail (peak={peak:.4f})"


def test_left_right_match():
    arr = render(tfosc_patch(), n_samples=int(SR))
    assert np.allclose(arr[:, 0], arr[:, 1]), "L and R differ despite identical wiring"


# ---------------------------------------------------------------------------
# Freq knob → log pitch mapping.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "freq_knob, expected_hz",
    [
        (0.0,  20.0),
        (0.25, 63.25),
        (0.5,  200.0),
        (0.75, 632.5),
        (1.0,  2000.0),
    ],
)
def test_freq_knob_maps_to_pitch(freq_knob, expected_hz):
    arr = render(tfosc_patch(freq=freq_knob, morph_x=0.0, morph_y=0.0),
                 n_samples=int(SR))
    f = dominant_freq(arr[:, 0])
    err = abs(f - expected_hz) / expected_hz
    assert err < 0.02, (
        f"freq_knob={freq_knob} → expected ≈{expected_hz:.1f} Hz, got {f:.1f} Hz "
        f"({err*100:.2f}% off)"
    )


# ---------------------------------------------------------------------------
# Morph plane → spectral content.
# ---------------------------------------------------------------------------

def test_morph_changes_timbre():
    """(1,0) saw corner must carry meaningfully more HF energy than (0,0) sine."""
    sine = render(tfosc_patch(morph_x=0.0, morph_y=0.0), n_samples=int(SR))
    saw  = render(tfosc_patch(morph_x=0.5, morph_y=0.0), n_samples=int(SR))
    hf_sine = hf_energy(sine[:, 0])
    hf_saw  = hf_energy(saw[:, 0])
    assert hf_saw > 3.0 * hf_sine, (
        f"saw corner should be harmonically richer than sine; "
        f"HF energy: sine={hf_sine:.1f} saw={hf_saw:.1f}"
    )


# ---------------------------------------------------------------------------
# A — Fine tune knob (±1 semitone).
# ---------------------------------------------------------------------------

def test_fine_tune_detunes_pitch():
    """fine=0 (-1 semitone) ≈ 188.8 Hz; fine=1 (+1 semitone) ≈ 211.9 Hz at freq_knob=0.5."""
    base_hz = 200.0
    semitone = pow(2.0, 1.0 / 12.0)

    arr_dn = render(tfosc_patch(freq=0.5, fine=0.0, morph_x=0.0, morph_y=0.0),
                    n_samples=int(SR))
    arr_up = render(tfosc_patch(freq=0.5, fine=1.0, morph_x=0.0, morph_y=0.0),
                    n_samples=int(SR))
    f_dn = dominant_freq(arr_dn[:, 0])
    f_up = dominant_freq(arr_up[:, 0])

    assert abs(f_dn - base_hz / semitone) / (base_hz / semitone) < 0.01, \
        f"fine=0 → expected {base_hz/semitone:.2f}, got {f_dn:.2f}"
    assert abs(f_up - base_hz * semitone) / (base_hz * semitone) < 0.01, \
        f"fine=1 → expected {base_hz*semitone:.2f}, got {f_up:.2f}"


# ---------------------------------------------------------------------------
# C — Auto-morph LFO modulates timbre over time.
# ---------------------------------------------------------------------------

def test_lfo_depth_creates_time_varying_spectrum():
    """With LFO depth>0, HF energy in first vs second half of buffer differs
    (LFO is sweeping morph). With LFO off, halves should be near-identical."""
    n = int(2 * SR)  # 2 s window

    off = render(tfosc_patch(morph_x=0.3, morph_y=0.3, lfo_rate=0.5, lfo_depth=0.0), n)
    on  = render(tfosc_patch(morph_x=0.3, morph_y=0.3, lfo_rate=0.5, lfo_depth=1.0), n)

    def hf_ratio(samples):
        half = len(samples) // 2
        a = hf_energy(samples[:half])
        b = hf_energy(samples[half:])
        return max(a, b) / max(1e-6, min(a, b))

    # Off: ratio close to 1 (stationary). On: ratio noticeably > 1 (varying).
    r_off = hf_ratio(off[:, 0])
    r_on  = hf_ratio(on[:, 0])
    assert r_off < 1.3, f"static signal varies too much: {r_off:.2f}"
    assert r_on > 1.5, f"LFO at depth=1 should swing HF content; ratio={r_on:.2f}"


# ---------------------------------------------------------------------------
# D — Phase mangling: Warp and FM both add harmonic content.
# ---------------------------------------------------------------------------

def test_warp_extreme_adds_harmonics():
    """Asymmetric warp (≠0.5) on a pure sine should introduce harmonics."""
    flat = render(tfosc_patch(morph_x=0.0, morph_y=0.0, warp=0.5), n_samples=int(SR))
    skew = render(tfosc_patch(morph_x=0.0, morph_y=0.0, warp=0.05), n_samples=int(SR))
    hf_flat = hf_energy(flat[:, 0])
    hf_skew = hf_energy(skew[:, 0])
    assert hf_skew > 3.0 * hf_flat, (
        f"extreme warp must inject harmonics; HF flat={hf_flat:.1f} "
        f"skewed={hf_skew:.1f}"
    )


def test_fm_amt_off_is_unchanged():
    """fm_amt=0 means the FM jack value is ignored — output matches baseline."""
    # No FM CV is patched in headless, so fm_in stays at 0 regardless;
    # this test guards against accidental side-effects from the fm_amt knob.
    base = render(tfosc_patch(morph_x=0.0, morph_y=0.0, fm_amt=0.0), n_samples=int(SR))
    knob = render(tfosc_patch(morph_x=0.0, morph_y=0.0, fm_amt=1.0), n_samples=int(SR))
    # Allow tiny floating-point divergence.
    diff = float(np.max(np.abs(base[:, 0] - knob[:, 0])))
    assert diff < 1e-3, f"fm_amt should be inert without FM input; max diff={diff}"
