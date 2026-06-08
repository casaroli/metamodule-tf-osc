---
name: tfexample-train
description: Retrain the TFExample SIREN wavetable-morph model and re-embed the resulting .tflite as a C array. Use when the user changes the anchor waveforms in models/train/train_siren_morph9.py, wants a fresh model, or asks to "retrain".
---

# Retrain the TFOsc SIREN model

Run the trainer, then convert the resulting `.tflite` into the embedded
C array that the plugin links against.

## Steps

1. **Run the trainer**. The venv must already exist at
   `models/train/.venv/`. If it doesn't, create it first:

   ```bash
   cd models/train
   python3 -m venv .venv
   .venv/bin/pip install --upgrade pip
   .venv/bin/pip install tensorflow scipy pytest
   ```

2. **Train**:

   ```bash
   cd models/train
   .venv/bin/python3 train_siren_morph9.py
   ```

   The script writes `models/wavetable_morph.tflite` (~41 KB float).
   It prints per-anchor correlations against ground truth at the end —
   anything below 0.95 is a problem.

3. **Re-embed as C array**:

   ```bash
   cd models
   xxd -i -n g_wavetable_morph_model_data wavetable_morph.tflite > .raw
   {
     echo '#include "wavetable_morph_model_data.h"'
     echo
     echo 'alignas(8) const unsigned char g_wavetable_morph_model_data[] = {'
     awk '/^unsigned char/{flag=1; next} /^unsigned int/{flag=0} flag && !/^};/' .raw
     echo '};'
     echo
     awk '/^unsigned int/{print "const unsigned int g_wavetable_morph_model_data_len = " $NF}' .raw
   } > wavetable_morph_model_data.cc
   rm .raw
   ```

4. **Check the op set** the trained model uses — it should be just
   `FULLY_CONNECTED` and `SIN`. Anything new means the runtime resolver
   needs `AddXxx()` calls (in `tf_osc.cc`'s constructor) and the
   corresponding kernel `.cc` file added to `tflite_micro` in
   `CMakeLists.txt`.

5. **Rebuild and test**:

   ```bash
   cd /Users/marco/ia/metamodule/metamodule/simulator
   cmake --build build
   cd tests
   ../models/train/.venv/bin/pytest -q
   ```

## When the user asks "did we retrain?"

The training scripts always overwrite `models/wavetable_morph.tflite`
with a fresh timestamp. Compare `ls -la` against the last edit time of
`tf_osc.cc` or related files. Mtime newer than the source = retrained.
