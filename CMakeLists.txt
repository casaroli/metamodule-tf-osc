cmake_minimum_required(VERSION 3.22)

if(NOT "${METAMODULE_SDK_DIR}" STREQUAL "")
    message("METAMODULE_SDK_DIR set by CMake variable ${METAMODULE_SDK_DIR}")
elseif(DEFINED ENV{METAMODULE_SDK_DIR})
    set(METAMODULE_SDK_DIR "$ENV{METAMODULE_SDK_DIR}")
    message("METAMODULE_SDK_DIR set by environment variable ${METAMODULE_SDK_DIR}")
else()
    set(METAMODULE_SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../metamodule-plugin-sdk")
    message("METAMODULE_SDK_DIR set to default: ${METAMODULE_SDK_DIR}")
endif()

include(${METAMODULE_SDK_DIR}/plugin.cmake)

project(TFExample
    VERSION 0.1
    DESCRIPTION "TFLite Micro audio-generation demo for MetaModule"
    LANGUAGES C CXX ASM
)

# ---------------------------------------------------------------------------
# TensorFlow Lite Micro — vendored slice built as a static library.
# ---------------------------------------------------------------------------
set(TFLM_DIR ${CMAKE_CURRENT_LIST_DIR}/third_party/tflite-micro)

add_library(tflite_micro STATIC
    # Core C/C++ runtime
    ${TFLM_DIR}/tensorflow/lite/core/c/common.cc
    ${TFLM_DIR}/tensorflow/lite/core/api/flatbuffer_conversions.cc
    ${TFLM_DIR}/tensorflow/lite/core/api/tensor_utils.cc

    # MLIR-side types/reporter/schema (newer upstream layout)
    ${TFLM_DIR}/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc
    ${TFLM_DIR}/tensorflow/compiler/mlir/lite/schema/schema_utils.cc

    # Reference kernels infrastructure
    ${TFLM_DIR}/tensorflow/lite/kernels/internal/common.cc
    ${TFLM_DIR}/tensorflow/lite/kernels/internal/portable_tensor_utils.cc
    ${TFLM_DIR}/tensorflow/lite/kernels/internal/quantization_util.cc
    ${TFLM_DIR}/tensorflow/lite/kernels/internal/runtime_shape.cc
    ${TFLM_DIR}/tensorflow/lite/kernels/internal/tensor_ctypes.cc
    ${TFLM_DIR}/tensorflow/lite/kernels/internal/tensor_utils.cc
    ${TFLM_DIR}/tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc
    ${TFLM_DIR}/tensorflow/lite/kernels/kernel_util.cc

    # Micro runtime
    ${TFLM_DIR}/tensorflow/lite/micro/debug_log.cc
    ${TFLM_DIR}/tensorflow/lite/micro/fake_micro_context.cc
    ${TFLM_DIR}/tensorflow/lite/micro/flatbuffer_utils.cc
    ${TFLM_DIR}/tensorflow/lite/micro/memory_helpers.cc
    ${TFLM_DIR}/tensorflow/lite/micro/micro_allocation_info.cc
    ${TFLM_DIR}/tensorflow/lite/micro/micro_allocator.cc
    ${TFLM_DIR}/tensorflow/lite/micro/micro_context.cc
    ${TFLM_DIR}/tensorflow/lite/micro/micro_interpreter.cc
    ${TFLM_DIR}/tensorflow/lite/micro/micro_interpreter_context.cc
    ${TFLM_DIR}/tensorflow/lite/micro/micro_interpreter_graph.cc
    ${TFLM_DIR}/tensorflow/lite/micro/micro_log.cc
    ${TFLM_DIR}/tensorflow/lite/micro/micro_op_resolver.cc
    ${TFLM_DIR}/tensorflow/lite/micro/micro_profiler.cc
    ${TFLM_DIR}/tensorflow/lite/micro/micro_resource_variable.cc
    ${TFLM_DIR}/tensorflow/lite/micro/micro_time.cc
    ${TFLM_DIR}/tensorflow/lite/micro/micro_utils.cc
    ${TFLM_DIR}/tensorflow/lite/micro/system_setup.cc
    ${TFLM_DIR}/tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cc
    ${TFLM_DIR}/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cc

    # Arena allocators
    ${TFLM_DIR}/tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cc
    ${TFLM_DIR}/tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cc
    ${TFLM_DIR}/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc

    # Memory planners
    ${TFLM_DIR}/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc
    ${TFLM_DIR}/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc
    ${TFLM_DIR}/tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.cc

    # The op kernels we use
    ${TFLM_DIR}/tensorflow/lite/micro/kernels/kernel_util.cc
    ${TFLM_DIR}/tensorflow/lite/micro/kernels/fully_connected.cc
    ${TFLM_DIR}/tensorflow/lite/micro/kernels/fully_connected_common.cc
    ${TFLM_DIR}/tensorflow/lite/micro/kernels/elementwise.cc  # provides SIN, COS, ABS, etc.
)

target_include_directories(tflite_micro PUBLIC
    ${TFLM_DIR}
    ${TFLM_DIR}/third_party/flatbuffers/include
    ${TFLM_DIR}/third_party/gemmlowp
    ${TFLM_DIR}/third_party/ruy
)

target_compile_definitions(tflite_micro PUBLIC
    TF_LITE_STATIC_MEMORY
    TF_LITE_DISABLE_X86_NEON
    TF_LITE_STRIP_ERROR_STRINGS
)

target_compile_options(tflite_micro PRIVATE
    -Wno-unused-parameter
    -Wno-sign-compare
    -Wno-missing-field-initializers
    -Wno-unused-variable
    -Wno-unused-function
    -Wno-type-limits
    -Wno-shadow
)

set_target_properties(tflite_micro PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# ---------------------------------------------------------------------------
# TFExample plugin proper
# ---------------------------------------------------------------------------
add_library(TFExample STATIC)
target_include_directories(TFExample PRIVATE .)

target_sources(TFExample PRIVATE
    tf_osc.cc
    tf_osc_elements.cc
    plugin.cc
    models/wavetable_morph_model_data.cc
)

set_target_properties(TFExample PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(TFExample PRIVATE tflite_micro)
set(EXTRA_LINK_OPTIONS $<TARGET_OBJECTS:tflite_micro>)

# Call this to link and create the plugin file
create_plugin(
    SOURCE_LIB      TFExample
    PLUGIN_NAME     TFExample
    PLUGIN_JSON     ${CMAKE_CURRENT_LIST_DIR}/plugin.json
    SOURCE_ASSETS   ${CMAKE_CURRENT_LIST_DIR}/assets
    DESTINATION     ${CMAKE_CURRENT_LIST_DIR}/../metamodule-plugins/
)
