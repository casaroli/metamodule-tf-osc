#pragma once

#include "CoreModules/CoreProcessor.hh"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <cstdint>

enum {
    FreqKnobID,
    FineKnobID,
    MorphXKnobID,
    MorphYKnobID,
    LfoRateKnobID,
    LfoDepthKnobID,
    WarpKnobID,
    FmAmtKnobID,
};
enum { VOctJackID, MorphXCvJackID, MorphYCvJackID, SyncJackID, FmInJackID };
enum { OutputJackID };
enum { ActLightID };

class TFOsc : public CoreProcessor {
public:
    TFOsc();

    void update() override;
    void set_param(int param_id, float val) override;
    void set_samplerate(float sr) override;
    void set_input(int input_id, float val) override;
    float get_output(int output_id) const override;
    float get_led_brightness(int led_id) const override;

private:
    // Arena sized for a 3→96→96→1 SIREN: ~768 B of activation tensors
    // plus TFLM bookkeeping. 16 KB leaves comfortable headroom.
    static constexpr int kArenaSize = 16384;
    static constexpr int kInferenceStride = 1;

    float sr_ = 48000.f;
    float phase_ = 0.f;
    float freq_knob_ = 0.f;
    float fine_knob_ = 0.5f;     // 0.5 = no detune; full range = ±1 semitone
    float morph_x_ = 0.f;        // knob
    float morph_y_ = 0.f;        // knob
    float morph_x_cv_ = 0.f;     // CV jack (volts)
    float morph_y_cv_ = 0.f;     // CV jack (volts)
    float v_oct_in_ = 0.f;
    float out_ = 0.f;

    // Internal auto-morph LFO — adds depth * sin(2π · lfo_phase) to Morph X.
    float lfo_rate_knob_ = 0.f;  // 0..1 → 0.05..5 Hz log
    float lfo_depth_ = 0.f;      // 0..1, scales the LFO before summing
    float lfo_phase_ = 0.f;

    // Hard sync: rising edge on the Sync jack snaps phase_ to 0.
    float prev_sync_in_ = 0.f;

    // D-flavor: phase mangling before the NN sees the phase.
    float warp_knob_ = 0.5f;     // 0.5 = identity; 0..1 = full L/R bias
    float fm_amt_knob_ = 0.f;    // 0..1 → 0..1 phase-add per volt
    float fm_in_ = 0.f;          // FM input jack (volts)

    uint32_t sample_counter_ = 0;
    float last_y_ = 0.f;
    float next_y_ = 0.f;

    alignas(16) uint8_t tensor_arena_[kArenaSize];
    tflite::MicroMutableOpResolver<2> resolver_;  // FullyConnected + Sin
    const tflite::Model* model_;
    tflite::MicroInterpreter interpreter_;
    TfLiteTensor* input_ = nullptr;
    TfLiteTensor* output_ = nullptr;
};
