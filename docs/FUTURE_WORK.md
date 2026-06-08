# Future Work

Things this experiment opened up but didn't pursue. Roughly ordered by
leverage / risk.

## Critical: Hardware validation

**Question we can't answer without a real MetaModule:** does per-sample
TFLM `Invoke()` at 48 kHz make audio budget on Cortex-A7?

Current model is a 3 → 96 → 96 → 1 SIREN (~10 K MACs/inference). At 48 kHz
that's 480 MMACs/s. Cortex-A7 @ 800 MHz NEON-VFPv4 peaks at ~3 GFLOPS, so
the optimistic load is ~16 %. Realistic load with overhead might be 25–60 %.

**Mitigation already in place:** `kInferenceStride` in `tf_osc.hh` is a
compile-time constant. Bumping to 4 / 8 / 16 cuts the inference rate
proportionally and the per-sample DSP linearly interpolates between
calls. At stride 16 the inference rate is 3 kHz, which is far inside any
plausible budget.

**Test plan:**
1. Build the `.mmplugin` for hardware.
2. Load it on a MetaModule.
3. Patch TFOsc → out, listen for clean output.
4. Check the on-device CPU meter under various morph + LFO + warp settings.
5. If pegged: bump `kInferenceStride`, rebuild, retest. Find the lowest
   stride that runs clean with all controls active.

If hardware turns out to barely fit, also consider:

- Quantize the SIREN to int8 (TFLite static post-training quantization).
  ~4× model size reduction and possible CMSIS-NN acceleration.
- Add a CMSIS-NN-backed FullyConnected kernel to the resolver
  (`AddFullyConnected(Register_FULLY_CONNECTED_INT8())` or similar).

## High leverage: model improvements

### Larger / richer anchor set

The current 3×3 grid is musically diverse but somewhat predictable. Ideas:

- 4×4 grid (16 anchors). Bilinear interp still over 4 nearest; the grid
  resolution increases. Trade: bigger model needed to keep correlation
  high. Likely 3 → 128 → 128 → 1 = ~70 KB tflite.
- Non-grid anchor placement (e.g. learned latent positions via a small
  autoencoder). Lets the timbral neighborhood reflect perceptual
  similarity rather than arbitrary 2D coordinates.
- Curated set including granular textures, drum-like transients, FM
  combinations, sub-octaves.

### Time-varying input

Right now the model is purely waveform-period: `(phase, mx, my) → sample`.
What if we also fed slow morph LFOs or pitch-bend trajectories so the
model could learn time-varying timbres (legato vowels, plucked decays,
etc.)? Requires a recurrent or stateful net — outside TFLM's current
sweet spot but possible with LSTM ops.

### Direct neural waveshaping

Skip the synthesis entirely and use the model as a learned distortion
curve on an external audio input. Input becomes `(audio_sample, mx, my)`,
output is the shaped sample. Different feature: not a synth voice but a
neural effect.

## Medium leverage: more synth controls

- **Sub-oscillator** — classical sine/square one octave below the NN
  output, summed in with its own mix knob. Adds bass weight.
- **Drive / saturation** — soft-clipping post-NN with its own knob.
- **One-pole lowpass filter** — single SVF on output. Closes the loop
  on "complete monosynth voice" feel.
- **AR or ADSR envelope** with a Gate input. Turns the module into a
  patchable instrument rather than a tone source. (Note: the user
  explicitly skipped this earlier; revisit if making it monosynth-y is
  the goal.)
- **Multi-voice unison** — run the NN N times per sample at slight
  detunes, sum, normalize. Super-saw style fatness. Costs N× inference.

## Medium leverage: shipping polish

- **Custom faceplate art** — current `tf_osc.png` is just the
  NativeExample asset reused. A proper panel design with labeled
  controls and a colored morph map would sell the module visually.
- **README assets** — a couple of spectrograms / waveform plots in the
  README to give the timbral palette some visual ground truth.
- **Demo patches that use the CV jacks** — `tfosc_test.yml` doesn't
  exercise MorphX/MorphY CV or FM In. Add a 2-3 module patch with a
  built-in `4msCompany:MultiLFO` patched into Morph X CV so loading the
  patch immediately shows the timbral plane sweeping.
- **Upstream the simulator patches** as a PR to `4ms/metamodule`. See
  [SIMULATOR_PATCHES.md](SIMULATOR_PATCHES.md).
- **CI** — GitHub Actions running the pytest suite on every push.

## Lower leverage: TFLM exploration

- **CMSIS-NN integrated build** — the `tflite-micro/tensorflow/lite/micro/kernels/cmsis_nn/`
  folder has optimized kernels for Cortex-M; for Cortex-A7 there are
  separate optimized paths. Worth measuring against the reference
  kernels we're using.
- **Quantization-aware training** — train the SIREN with `tf.quantization.fake_quant`
  to retain quality after int8 conversion.
- **Multi-model** — the plugin could hold N models and switch between
  them (e.g. wavetable for osc, distortion curve for waveshaper, IR
  for reverb). Each gets its own interpreter + arena.

## Other directions

- **Polyphony.** Replicate TFOsc N times in a single module
  with a polyphony selector knob; share the model + interpreter
  (instance allocations are cheap relative to the arena).
- **Wave-folder integration.** Post-process the model output through a
  classical wavefolder. Cheap, dramatic timbral expansion.
- **Train the model on hardware-recorded analog waveforms.** Sample
  classic analog oscillators (Moog, Roland, etc.), use them as anchor
  waveforms instead of synthesized math. The neural morph plane
  becomes a "morph between analog oscillator characters" — genuinely
  novel and not easy to do with classical wavetable.
- **Train on someone's instrument samples.** Sample a piano, vocal,
  guitar, brass at one pitch each, use as anchors. The morph plane
  becomes "blend between four real instruments."
- **A second module: TFFilter.** Same TFLM-pipeline idea but for a
  learned filter — input `(audio_sample, freq, resonance)` → filtered
  sample. Would showcase TFLM in a non-generator role.

## What I'd pick first

In order of expected impact-per-hour:

1. **Hardware validation** — answers the only existential question.
2. **Polish patches + upstream simulator changes** — turns this from
   "personal project" into "shareable example for the next person."
3. **Train on real recorded analog waveforms** — actual novelty,
   demonstrates the "neural synth" pitch beyond toy waveforms.
4. **Multi-voice unison + sub-osc** — biggest sonic improvement for
   smallest code surface.

The rest are interesting but speculative until #1 is resolved.
