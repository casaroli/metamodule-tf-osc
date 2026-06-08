# Simulator-side patches

To make TFExample run inside the MetaModule simulator (both GUI and
headless), four local changes were applied to a clone of
[4ms/metamodule](https://github.com/4ms/metamodule). They are documented
here so they can be either applied manually or eventually upstreamed as
a PR.

All paths are relative to the simulator repo root
(`metamodule/`).

## 1. Register TFExample as an ext-builtin plugin

**`simulator/ext-plugins.cmake`** — add three lines near the top:

```cmake
list(APPEND ext_builtin_brand_paths   "${CMAKE_CURRENT_LIST_DIR}/../../metamodule-plugin-examples/TFExample")
list(APPEND ext_builtin_brand_libname "TFExample")
list(APPEND ext_builtin_brand_slug    "TFExample")
```

This causes `add_subdirectory(...)` of TFExample, which builds it as a
static library (`libTFExample.a`) and codegens the init call into
`build/ext_plugin/ext_plugin_builtin.hh`:

```cpp
inline void load_ext_builtin_plugins(auto &internal_plugins) {
    extern void init_TFExample(rack::plugin::Plugin *);
    init_TFExample(&internal_plugins.emplace_back("TFExample"));
}
```

## 2. Headless main: call `load_ext_builtin_plugins()`

The GUI path constructs an `InternalPluginManager` (in
`firmware/vcv_ports/internal_plugin_manager.hh`) which itself calls
`load_ext_builtin_plugins()`. The headless main never instantiates that
manager, so the brand never registers. Fix in
**`simulator/src/headless/main-headless.cc`**:

```diff
+#include "plugin/Plugin.hpp"
+#include "ext_plugin_builtin.hh"
+
 #include "audio_files.hh"
 #include "audio_wrapper.hh"
 #include "file_io.hh"
 #include "patch-serial/yaml_to_patch.hh"
 #include "settings.hh"
 #include <chrono>
+#include <list>
 #include <span>
```

And in `main()`, before `read_patch()`:

```cpp
std::list<rack::plugin::Plugin> ext_plugins;
load_ext_builtin_plugins(ext_plugins);
```

**Include order matters.** `plugin/Plugin.hpp` must come before
`ext_plugin_builtin.hh`, otherwise the `Jack` struct from `patch-serial`
shadows the `rack::plugin::` namespace lookup and the build fails with
"no member named 'plugin' in 'Jack'".

## 3. Always link ThorVG (drop the headless gate)

`firmware/coreproc_plugin/graphics/waveform_display.cc` references
`tvg::SwCanvas` symbols unconditionally, but the simulator's `CMakeLists.txt`
and the firmware's `coreproc_plugin/CMakeLists.txt` only link ThorVG
when `HEADLESS=OFF`. Headless link fails with undefined ThorVG symbols.

Fix in **`simulator/CMakeLists.txt`** — remove the `if (NOT HEADLESS)`
guards around the ThorVG `add_subdirectory` and `target_link_libraries`
entries:

```diff
 # #################### ThorVG ############################################

-if (NOT HEADLESS)
-    add_subdirectory(${FWDIR}/lib/thorvg ${CMAKE_CURRENT_BINARY_DIR}/thorvg)
-endif()
+add_subdirectory(${FWDIR}/lib/thorvg ${CMAKE_CURRENT_BINARY_DIR}/thorvg)
```

```diff
-if (NOT HEADLESS)
-    target_link_libraries(CoreModules-4ms PRIVATE ThorVG)
-endif()
+target_link_libraries(CoreModules-4ms PRIVATE ThorVG)
```

And move `ThorVG` out of the GUI-only `target_link_libraries(simulator ...)`
block into the unconditional one.

Mirror the same change in **`firmware/coreproc_plugin/CMakeLists.txt`**:

```diff
-if (NOT HEADLESS)
-    target_link_libraries(coreproc_plugin_export PRIVATE ThorVG)
-endif()
+target_link_libraries(coreproc_plugin_export PRIVATE ThorVG)
```

Cost: ~few hundred KB extra in the headless binary. Headless wins.

## 4. Stubs for FontStash, NanoVG pixbuf, and file-browser

The vcv_plugin/export code path always references:

- `fonsCreateInternal()`, `fonsAddFont()` — from
  `firmware/vcv_plugin/export/src/context.cc`
- `nvgCreatePixelBufferContext()`, `nvgDeletePixelBufferContext()` —
  from `firmware/vcv_plugin/export/src/engine/Module.cpp`
- `MetaModule::show_file_browser()`, `MetaModule::show_file_save_dialog()` —
  from various firmware GUI code

In GUI builds these are satisfied by `firmware/vcv_plugin/internal/fons-wrapper.cc`,
`nanovg_pixbuf.cc`, and `simulator/src/file_browser_adaptor.cc`. All of
those pull lvgl into the headless build, which we don't want.

Add two new stub files:

**`simulator/stubs/fons_stubs.cc`** (~25 lines):

```cpp
#include "vcv_plugin/export/nanovg/fontstash-wrapper.h"
#include <cstdint>
#include <span>

struct NVGcontext;

FONScontext* fonsCreateInternal() { return nullptr; }
int fonsAddFont(FONScontext*, const char*, const char*, int) { return -1; }

NVGcontext* nvgCreatePixelBufferContext(void*, std::span<uint32_t>,
                                        uint32_t, uint32_t) { return nullptr; }
void nvgDeletePixelBufferContext(NVGcontext*) {}
```

**`simulator/stubs/file_browser_stub.cc`** (~15 lines):

```cpp
#include "gui/pages/file_browser/file_browser_adaptor.hh"

namespace MetaModule {
void show_file_browser(FileBrowserDialog*, std::string_view, std::string_view,
                       std::string_view, const std::function<void(char*)>) {}
void show_file_save_dialog(FileSaveDialog*, std::string_view, std::string_view,
                           std::string_view, std::function<void(char*)>) {}
}
```

Wire them into the headless target in `simulator/CMakeLists.txt`:

```diff
 if(HEADLESS)
     add_executable(simulator
         src/headless/main-headless.cc
         src/headless/audio_files.cc
         src/headless/nanovg.cc
         stubs/svg.cc
+        stubs/fons_stubs.cc
+        stubs/async_thread_control.cc
+        stubs/file_browser_stub.cc
         hardware_support/random.cc
         ...
+        ${FWDIR}/src/patch_play/plugin_module.cc
+        ${FWDIR}/coreproc_plugin/internal_interface/plugin_app_interface.cc
         ${FWDIR}/metamodule-plugin-sdk/version.cc
     )
```

And add the include dirs the firmware sources need
(`${FWDIR}/src/fs/fatfs`, `${FWDIR}/lib/fatfs/source`, `${FWDIR}/src/params`)
to the unconditional `target_include_directories(simulator ...)` block.

## Test the patches

After applying:

```bash
cd metamodule/simulator
cmake --fresh --preset headless-min
cmake --build build
./build/simulator -p patches/tfosc_test.yml -n 48000 -o /tmp/out.raw
```

Should print "Patch loaded: 2 modules" and produce a non-silent WAV at
`/tmp/out.raw` (despite the `.raw` extension — the header is RIFF/WAVE).

## Upstreaming

These are general-utility changes that any custom-plugin author would
need. A clean PR to `4ms/metamodule` should:

1. Open a single PR with all four changes.
2. In `ext-plugins.cmake`, remove the project-specific
   `ext_builtin_brand_paths` line — it's a per-user customization, not
   something to merge.
3. Keep the headless-build fixes general (no TFExample-specific code).
4. Stub files go in `simulator/stubs/` next to the existing `svg.cc`.

Open questions for a maintainer:

- Should the stubs be header-only inline implementations instead of
  separate `.cc` files?
- Should the headless build keep linking ThorVG (cheaper) or do a
  proper code-level split that lets `waveform_display.cc` compile to a
  no-op in headless mode (cleaner)?
- Should `load_ext_builtin_plugins()` be called inside a higher-level
  setup routine that both GUI and headless mains share?

Not blocking for the experiment, but worth raising at PR time.
