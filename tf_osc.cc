#include "tf_osc.hh"
#include "models/wavetable_morph_model_data.h"

#include <cmath>

TFOsc::TFOsc()
    : tensor_arena_{}
    , resolver_()
    , model_(tflite::GetModel(g_wavetable_morph_model_data))
    , interpreter_(model_, resolver_, tensor_arena_, kArenaSize) {
    (void)resolver_.AddFullyConnected();
    (void)resolver_.AddSin();
    if (interpreter_.AllocateTensors() == kTfLiteOk) {
        input_ = interpreter_.input(0);
        output_ = interpreter_.output(0);
    }
}

void TFOsc::set_samplerate(float sr) {
    sr_ = sr > 0.f ? sr : 48000.f;
}

void TFOsc::set_param(int param_id, float val) {
    switch (param_id) {
        case FreqKnobID:     freq_knob_ = val;     break;
        case FineKnobID:     fine_knob_ = val;     break;
        case MorphXKnobID:   morph_x_ = val;       break;
        case MorphYKnobID:   morph_y_ = val;       break;
        case LfoRateKnobID:  lfo_rate_knob_ = val; break;
        case LfoDepthKnobID: lfo_depth_ = val;     break;
        case WarpKnobID:     warp_knob_ = val;     break;
        case FmAmtKnobID:    fm_amt_knob_ = val;   break;
        default: break;
    }
}

void TFOsc::set_input(int input_id, float val) {
    switch (input_id) {
        case VOctJackID:     v_oct_in_ = val; break;
        case MorphXCvJackID: morph_x_cv_ = val; break;
        case MorphYCvJackID: morph_y_cv_ = val; break;
        case SyncJackID: {
            // Rising edge through ~1V → reset phase. Compare hysteresis-style
            // to ignore noise around the threshold.
            if (prev_sync_in_ < 1.f && val >= 1.f) {
                phase_ = 0.f;
            }
            prev_sync_in_ = val;
            break;
        }
        case FmInJackID: fm_in_ = val; break;
        default: break;
    }
}

void TFOsc::update() {
    // Pitch: base 20..2000 Hz log, ±1 semitone fine, V/Oct shifts octaves.
    const float base_hz = 20.f * powf(100.f, freq_knob_);
    const float fine_semitones = (fine_knob_ - 0.5f) * 2.f;       // ±1
    const float pitch_hz = base_hz
        * powf(2.f, v_oct_in_)
        * powf(2.f, fine_semitones / 12.f);
    phase_ += pitch_hz / sr_;
    if (phase_ >= 1.f) phase_ -= 1.f;

    if (input_ == nullptr || output_ == nullptr) {
        out_ = 0.f;
        return;
    }

    if ((sample_counter_++ % kInferenceStride) == 0) {
        last_y_ = next_y_;

        // Auto-morph LFO: 0.05..5 Hz log, advances every sample.
        const float lfo_hz = 0.05f * powf(100.f, lfo_rate_knob_);
        lfo_phase_ += lfo_hz / sr_;
        if (lfo_phase_ >= 1.f) lfo_phase_ -= 1.f;
        const float lfo_val = sinf(lfo_phase_ * 6.28318530717958f) * lfo_depth_;

        // CV jacks sum into the knob value at ±5V → ±1 morph; LFO adds to X.
        float mx = morph_x_ + morph_x_cv_ * 0.2f + lfo_val;
        float my = morph_y_ + morph_y_cv_ * 0.2f;
        if (mx < 0.f) mx = 0.f; else if (mx > 1.f) mx = 1.f;
        if (my < 0.f) my = 0.f; else if (my > 1.f) my = 1.f;

        // ---- D: phase mangling ----
        // 1) FM: add (fm_in_volts * fm_amt) / 5  → ±fm_amt phase units per ±5V.
        float ph = phase_ + (fm_in_ * fm_amt_knob_) * 0.2f;
        ph -= floorf(ph);  // wrap into [0, 1)
        // 2) Phase distortion (Casio PD piecewise linear, controlled by warp).
        //    warp=0.5 → identity; warp<0.5 squeezes phase toward early portion;
        //    warp>0.5 squeezes toward late portion. Creates asymmetric / PWM-like
        //    waveshape variations the NN was not directly trained on.
        const float w = (warp_knob_ < 0.001f) ? 0.001f
                      : (warp_knob_ > 0.999f) ? 0.999f
                      : warp_knob_;
        if (ph < w)
            ph = ph * 0.5f / w;
        else
            ph = 0.5f + (ph - w) * 0.5f / (1.f - w);

        // Model expects (phase ∈ [0,1), morph_x ∈ [0,1], morph_y ∈ [0,1])
        // and returns a single sample roughly in [-1, +1].
        input_->data.f[0] = ph;
        input_->data.f[1] = mx;
        input_->data.f[2] = my;
        (void)interpreter_.Invoke();
        next_y_ = output_->data.f[0];
    }

    float y;
    if constexpr (kInferenceStride > 1) {
        const float frac = (float)(sample_counter_ % kInferenceStride) / (float)kInferenceStride;
        y = last_y_ + frac * (next_y_ - last_y_);
    } else {
        y = next_y_;
    }

    if (y >  1.f) y =  1.f;
    if (y < -1.f) y = -1.f;
    out_ = y * 5.f;
}

float TFOsc::get_output(int output_id) const {
    if (output_id == OutputJackID)
        return out_;
    return 0.f;
}

float TFOsc::get_led_brightness(int led_id) const {
    if (led_id == ActLightID)
        return fabsf(out_) / 5.f;
    return 0.f;
}
