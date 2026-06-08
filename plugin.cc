
void init_tf_osc();

// Hardware plugin loader looks up the symbol "init".
extern "C" void init() {
  init_tf_osc();
}

// Simulator's ext-plugins.cmake codegens
//   extern void init_<brand>(rack::plugin::Plugin*);
// and statically links many plugins into one binary, so the per-plugin
// init symbol must match the brand slug and take a (discarded) Plugin*.
namespace rack { namespace plugin { struct Plugin; } }
void init_TFExample(rack::plugin::Plugin*) {
  init_tf_osc();
}
