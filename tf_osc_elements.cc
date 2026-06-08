#include "CoreModules/elements/element_counter.hh"
#include "CoreModules/elements/elements.hh"
#include "CoreModules/register_module.hh"
#include "tf_osc.hh"

void init_tf_osc() {
    // 8 knobs in 4 rows of 2, then 6 jacks in 2 rows of 3, plus an LED.
    // Panel is 14 HP wide (≈ 71 mm) so nothing clips.
    static std::array<MetaModule::Element, 15> elements;
    static std::array<ElementCount::Indices, 15> indices;

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

    // Two knob columns at x = 22 and x = 49, four rows from y = 12 down.
    elements[0]  = knob(22, 12, "Freq");        indices[0]  = {.param_idx = FreqKnobID};
    elements[1]  = knob(49, 12, "Fine");        indices[1]  = {.param_idx = FineKnobID};

    elements[2]  = knob(22, 30, "Morph X");     indices[2]  = {.param_idx = MorphXKnobID};
    elements[3]  = knob(49, 30, "Morph Y");     indices[3]  = {.param_idx = MorphYKnobID};

    elements[4]  = knob(22, 48, "Warp");        indices[4]  = {.param_idx = WarpKnobID};
    elements[5]  = knob(49, 48, "FM Amt");      indices[5]  = {.param_idx = FmAmtKnobID};

    elements[6]  = knob(22, 66, "LFO Rate");    indices[6]  = {.param_idx = LfoRateKnobID};
    elements[7]  = knob(49, 66, "LFO Depth");   indices[7]  = {.param_idx = LfoDepthKnobID};

    // Jack rows at y = 84 and y = 100. Three columns at x = 12, 35, 58.
    elements[8]  = in_jack(12, 84, "V/Oct");    indices[8]  = {.input_idx = VOctJackID};
    elements[9]  = in_jack(35, 84, "Sync");     indices[9]  = {.input_idx = SyncJackID};
    elements[10] = in_jack(58, 84, "FM In");    indices[10] = {.input_idx = FmInJackID};

    elements[11] = in_jack(12, 100, "X CV");    indices[11] = {.input_idx = MorphXCvJackID};
    elements[12] = in_jack(35, 100, "Y CV");    indices[12] = {.input_idx = MorphYCvJackID};
    elements[13] = out_jack(58, 100, "Out");    indices[13] = {.output_idx = OutputJackID};

    // LED tucked above the first jack row, centered.
    MetaModule::MonoLight light;
    light.x_mm = 35; light.y_mm = 74;
    light.image = "TFExample/components/led.png";
    light.short_name = "Activity";
    elements[14] = light; indices[14] = {.light_idx = ActLightID};

    MetaModule::ModuleInfoView info{
        .description = "TFLite Micro neural wavetable morph",
        .width_hp = 14,
        .elements = elements,
        .indices = indices,
    };

    MetaModule::register_module<TFOsc>("TFExample", "TFOsc", info,
                                       "TFExample/tf_osc.png");
}
