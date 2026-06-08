#include "CoreModules/elements/element_counter.hh"
#include "CoreModules/elements/elements.hh"
#include "CoreModules/register_module.hh"
#include "tf_osc.hh"

void init_tf_osc() {
    static std::array<MetaModule::Element, 16> elements;
    static std::array<ElementCount::Indices, 16> indices;

    auto knob = [](float x, float y, const char* name) {
        MetaModule::Knob k;
        k.x_mm = x; k.y_mm = y;
        k.image = "TFExample/components/knob.png";
        k.short_name = name;
        return k;
    };
    auto in_jack = [](float x, float y, const char* name) {
        MetaModule::JackInput j;
        j.x_mm = x; j.y_mm = y;
        j.image = "TFExample/components/jack.png";
        j.short_name = name;
        return j;
    };
    auto out_jack = [](float x, float y, const char* name) {
        MetaModule::JackOutput j;
        j.x_mm = x; j.y_mm = y;
        j.image = "TFExample/components/jack.png";
        j.short_name = name;
        return j;
    };

    // 10HP wide (≈50 mm). Two columns at x=15 / x=35.
    elements[0]  = knob(15, 10, "Freq");        indices[0]  = {.param_idx = FreqKnobID};
    elements[1]  = knob(35, 10, "Fine");        indices[1]  = {.param_idx = FineKnobID};

    elements[2]  = knob(15, 25, "Morph X");     indices[2]  = {.param_idx = MorphXKnobID};
    elements[3]  = knob(35, 25, "Morph Y");     indices[3]  = {.param_idx = MorphYKnobID};

    elements[4]  = knob(15, 40, "Warp");        indices[4]  = {.param_idx = WarpKnobID};
    elements[5]  = knob(35, 40, "FM Amt");      indices[5]  = {.param_idx = FmAmtKnobID};

    elements[6]  = knob(15, 55, "LFO Rate");    indices[6]  = {.param_idx = LfoRateKnobID};
    elements[7]  = knob(35, 55, "LFO Depth");   indices[7]  = {.param_idx = LfoDepthKnobID};

    // Jacks row 1 (CV inputs).
    elements[8]  = in_jack(10, 72, "X CV");     indices[8]  = {.input_idx = MorphXCvJackID};
    elements[9]  = in_jack(25, 72, "Y CV");     indices[9]  = {.input_idx = MorphYCvJackID};
    elements[10] = in_jack(40, 72, "FM In");    indices[10] = {.input_idx = FmInJackID};

    // Jacks row 2 (gates / V/oct).
    elements[11] = in_jack(10, 87, "V/Oct");    indices[11] = {.input_idx = VOctJackID};
    elements[12] = in_jack(25, 87, "Sync");     indices[12] = {.input_idx = SyncJackID};
    elements[13] = out_jack(40, 87, "Out");     indices[13] = {.output_idx = OutputJackID};

    MetaModule::MonoLight light;
    light.x_mm = 25; light.y_mm = 100;
    light.image = "TFExample/components/led.png";
    light.short_name = "Activity";
    elements[14] = light; indices[14] = {.light_idx = ActLightID};

    // Padding slot (unused element) so the array size matches indices.
    elements[15] = MetaModule::Knob{}; indices[15] = {};

    MetaModule::ModuleInfoView info{
        .description = "TFLite Micro neural wavetable morph",
        .width_hp = 10,
        .elements = std::span{elements.data(), 15},
        .indices = std::span{indices.data(), 15},
    };

    MetaModule::register_module<TFOsc>("TFExample", "TFOsc", info,
                                       "TFExample/tf_osc.png");
}
