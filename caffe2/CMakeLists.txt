# ---[ Generate and install header and cpp files
include(../cmake/Codegen.cmake)

# ---[ Vulkan code gen
if(USE_VULKAN)
  include(../cmake/VulkanCodegen.cmake)
endif()

# Debug messages - if you want to get a list of source files and examine
# target information, enable the following by -DPRINT_CMAKE_DEBUG_INFO=ON.
set(PRINT_CMAKE_DEBUG_INFO FALSE CACHE BOOL "print cmake debug information")
if(PRINT_CMAKE_DEBUG_INFO)
  include(../cmake/DebugHelper.cmake)
endif()

# ATen parallelism settings
#  OMP - OpenMP for intra-op, native thread pool for inter-op parallelism
#  NATIVE - using native thread pool for intra- and inter-op parallelism
if(INTERN_BUILD_MOBILE)
  set(ATEN_THREADING "NATIVE" CACHE STRING "ATen parallel backend")
else()
  if(USE_OPENMP)
    set(ATEN_THREADING "OMP" CACHE STRING "ATen parallel backend")
  else()
    set(ATEN_THREADING "NATIVE" CACHE STRING "ATen parallel backend")
  endif()
endif()

set(AT_PARALLEL_OPENMP 0)
set(AT_PARALLEL_NATIVE 0)

message(STATUS "Using ATen parallel backend: ${ATEN_THREADING}")
if("${ATEN_THREADING}" STREQUAL "OMP")
  set(AT_PARALLEL_OPENMP 1)
elseif("${ATEN_THREADING}" STREQUAL "NATIVE")
  set(AT_PARALLEL_NATIVE 1)
else()
  message(FATAL_ERROR "Unknown ATen parallel backend: ${ATEN_THREADING}")
endif()

# ---[ Declare source file lists

# ---[ ATen build
if(INTERN_BUILD_ATEN_OPS)
  set(__torch_CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE})
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_subdirectory(../aten aten)
  set(CMAKE_POSITION_INDEPENDENT_CODE ${__torch_CMAKE_POSITION_INDEPENDENT_CODE})

  # Generate the headers wrapped by our operator
  file(GLOB_RECURSE torchgen_python "${PROJECT_SOURCE_DIR}/torchgen/*.py")


  # Add source, includes, and libs to lists
  list(APPEND Caffe2_CPU_SRCS ${ATen_CPU_SRCS})
  list(APPEND Caffe2_GPU_SRCS ${ATen_CUDA_CPP_SRCS})
  list(APPEND Caffe2_XPU_SRCS ${ATen_XPU_SRCS})
  list(APPEND Caffe2_XPU_INCLUDE ${ATen_XPU_INCLUDE})
  list(APPEND Caffe2_XPU_DEPENDENCY_LIBS ${ATen_XPU_DEPENDENCY_LIBS})
  list(APPEND Caffe2_GPU_SRCS_W_SORT_BY_KEY ${ATen_CUDA_SRCS_W_SORT_BY_KEY})
  list(APPEND Caffe2_GPU_CU_SRCS ${ATen_CUDA_CU_SRCS})
  list(APPEND Caffe2_GPU_CU_SRCS_W_SORT_BY_KEY ${ATen_CUDA_CU_SRCS_W_SORT_BY_KEY})
  list(APPEND Caffe2_HIP_SRCS ${ATen_HIP_SRCS})
  list(APPEND Caffe2_MPS_SRCS ${ATen_MPS_SRCS})
  list(APPEND Caffe2_XPU_SRCS ${ATen_XPU_SRCS})
  list(APPEND Caffe2_HIP_SRCS ${ATen_HIP_SRCS_W_SORT_BY_KEY})
  list(APPEND Caffe2_CPU_TEST_SRCS ${ATen_CPU_TEST_SRCS})
  list(APPEND Caffe2_MPS_TEST_SRCS ${ATen_MPS_TEST_SRCS})
  list(APPEND Caffe2_GPU_TEST_SRCS ${ATen_CUDA_TEST_SRCS})
  list(APPEND Caffe2_HIP_TEST_SRCS ${ATen_HIP_TEST_SRCS})
  list(APPEND Caffe2_XPU_TEST_SRCS ${ATen_XPU_TEST_SRCS})
  list(APPEND Caffe2_CPU_TEST_SRCS ${ATen_CORE_TEST_SRCS})
  list(APPEND Caffe2_VULKAN_TEST_SRCS ${ATen_VULKAN_TEST_SRCS})
  list(APPEND Caffe2_CPU_INCLUDE ${ATen_CPU_INCLUDE})
  list(APPEND Caffe2_GPU_INCLUDE ${ATen_CUDA_INCLUDE})
  list(APPEND Caffe2_HIP_INCLUDE ${ATen_HIP_INCLUDE})
  list(APPEND Caffe2_XPU_INCLUDE ${ATen_XPU_INCLUDE})
  list(APPEND Caffe2_VULKAN_INCLUDE ${ATen_VULKAN_INCLUDE})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${ATen_CPU_DEPENDENCY_LIBS})
  list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS ${ATen_CUDA_DEPENDENCY_LIBS})
  list(APPEND Caffe2_HIP_DEPENDENCY_LIBS ${ATen_HIP_DEPENDENCY_LIBS})
  list(APPEND Caffe2_DEPENDENCY_INCLUDE ${ATen_THIRD_PARTY_INCLUDE})
  set(Caffe2_CUDA_DEPENDENCY_LIBS ${Caffe2_CUDA_DEPENDENCY_LIBS} PARENT_SCOPE)
endif()

# ---[ Caffe2 build
# Note: the folders that are being commented out have not been properly
# addressed yet.

if(NOT MSVC AND USE_XNNPACK)
  if(NOT TARGET fxdiv)
    set(FXDIV_BUILD_TESTS OFF CACHE BOOL "")
    set(FXDIV_BUILD_BENCHMARKS OFF CACHE BOOL "")
    add_subdirectory(
      "${FXDIV_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/FXdiv")
  endif()
endif()

add_subdirectory(core)
add_subdirectory(serialize)
add_subdirectory(utils)
if(NOT USE_FBGEMM)
  add_subdirectory(perfkernels)
endif()

# Advanced: if we have allow list specified, we will do intersections for all
# main lib srcs.
if(CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_CPU_SRCS CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_GPU_SRCS CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_XPU_SRCS CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_GPU_SRCS_W_SORT_BY_KEY CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_GPU_CU_SRCS CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_GPU_CU_SRCS_W_SORT_BY_KEY CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_HIP_SRCS CAFFE2_ALLOWLISTED_FILES)
endif()

if(PRINT_CMAKE_DEBUG_INFO)
  message(STATUS "CPU sources: ")
  foreach(tmp ${Caffe2_CPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU sources: (for torch_cuda_cpp)")
  foreach(tmp ${Caffe2_GPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU sources: (for torch_cuda_cu)")
  foreach(tmp ${Caffe2_GPU_CU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "torch_cuda_cu GPU sources (w/ sort by key): ")
  foreach(tmp ${Caffe2_GPU_CU_SRCS_W_SORT_BY_KEY})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "torch_cuda_cpp GPU sources (w/ sort by key): ")
  foreach(tmp ${Caffe2_GPU_SRCS_W_SORT_BY_KEY})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "CPU include: ")
  foreach(tmp ${Caffe2_CPU_INCLUDE})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU include: ")
  foreach(tmp ${Caffe2_GPU_INCLUDE})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "CPU test sources: ")
  foreach(tmp ${Caffe2_CPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU test sources: ")
  foreach(tmp ${Caffe2_GPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "HIP sources: ")
  foreach(tmp ${Caffe2_HIP_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "MPS sources: ")
  foreach(tmp ${Caffe2_MPS_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "XPU sources: ")
  foreach(tmp ${Caffe2_XPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "HIP test sources: ")
  foreach(tmp ${Caffe2_HIP_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen CPU test sources: ")
  foreach(tmp ${ATen_CPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen MPS test sources: ")
  foreach(tmp ${ATen_MPS_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen CUDA test sources: ")
  foreach(tmp ${ATen_CUDA_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen HIP test sources: ")
  foreach(tmp ${ATen_HIP_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen XPU test sources: ")
  foreach(tmp ${ATen_XPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen Vulkan test sources: ")
  foreach(tmp ${ATen_VULKAN_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

endif()

# ==========================================================
# formerly-libtorch
# ==========================================================

set(TORCH_SRC_DIR "${PROJECT_SOURCE_DIR}/torch")
set(TORCH_ROOT "${PROJECT_SOURCE_DIR}")

if(NOT TORCH_INSTALL_BIN_DIR)
  set(TORCH_INSTALL_BIN_DIR bin)
endif()

if(NOT TORCH_INSTALL_INCLUDE_DIR)
  set(TORCH_INSTALL_INCLUDE_DIR include)
endif()

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# Generate files
set(TOOLS_PATH "${TORCH_ROOT}/tools")

configure_file("${TORCH_SRC_DIR}/_utils_internal.py"
  "${TOOLS_PATH}/shared/_utils_internal.py"
  COPYONLY)

# Generate header with version info
configure_file("${TORCH_SRC_DIR}/csrc/api/include/torch/version.h.in"
  "${TORCH_SRC_DIR}/csrc/api/include/torch/version.h"
  @ONLY)

set(GENERATED_CXX_TORCH
  "${TORCH_SRC_DIR}/csrc/autograd/generated/Functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/ViewFuncs.cpp"
  )

if(NOT INTERN_DISABLE_AUTOGRAD AND NOT BUILD_LITE_INTERPRETER)
  list(APPEND GENERATED_CXX_TORCH
    "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_0.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_1.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_2.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_3.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_4.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/TraceType_0.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/TraceType_1.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/TraceType_2.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/TraceType_3.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/TraceType_4.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/ADInplaceOrViewType_0.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/ADInplaceOrViewType_1.cpp"
    "${TORCH_SRC_DIR}/csrc/inductor/aoti_torch/generated/c_shim_cpu.cpp"
  )
  if(BUILD_LAZY_TS_BACKEND)
    list(APPEND GENERATED_CXX_TORCH
      "${TORCH_SRC_DIR}/csrc/lazy/generated/LazyNativeFunctions.cpp"
      "${TORCH_SRC_DIR}/csrc/lazy/generated/RegisterAutogradLazy.cpp"
      "${TORCH_SRC_DIR}/csrc/lazy/generated/RegisterLazy.cpp"
    )
  endif()
endif()

set(GENERATED_H_TORCH
  "${TORCH_SRC_DIR}/csrc/autograd/generated/Functions.h"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/variable_factories.h"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/ViewFuncs.h"
  )

if(NOT INTERN_DISABLE_AUTOGRAD)
  list(APPEND GENERATED_H_TORCH
    "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType.h"
    "${TORCH_SRC_DIR}/csrc/lazy/generated/LazyIr.h"
    "${TORCH_SRC_DIR}/csrc/lazy/generated/LazyNonNativeIr.h"
    "${TORCH_SRC_DIR}/csrc/lazy/generated/LazyNativeFunctions.h"
  )
endif()

set(GENERATED_CXX_PYTHON
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions_0.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions_1.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions_2.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions_3.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions_4.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_variable_methods.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_torch_functions_0.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_torch_functions_1.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_torch_functions_2.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_nn_functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_fft_functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_linalg_functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_nested_functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_sparse_functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_special_functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_return_types.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_enum_tag.cpp"
  )

set(GENERATED_H_PYTHON
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions.h"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_return_types.h"
  )

set(GENERATED_TESTING_PYTHON
  "${TORCH_SRC_DIR}/testing/_internal/generated/annotated_fn_args.py"
  )

set(GENERATED_CXX_TORCH_CUDA
  "${TORCH_SRC_DIR}/csrc/inductor/aoti_torch/generated/c_shim_cuda.cpp"
  )

set(TORCH_GENERATED_CODE
  ${GENERATED_CXX_TORCH}
  ${GENERATED_H_TORCH}
  ${GENERATED_CXX_PYTHON}
  ${GENERATED_H_PYTHON}
  ${GENERATED_TESTING_PYTHON}
  ${GENERATED_CXX_TORCH_CUDA}
  )

set(GEN_PER_OPERATOR_FLAG)
if(USE_PER_OPERATOR_HEADERS)
  list(APPEND GEN_PER_OPERATOR_FLAG "--per_operator_headers")
endif()

file(GLOB_RECURSE autograd_python "${TOOLS_PATH}/autograd/*.py")
file(GLOB_RECURSE autograd_yaml "${TOOLS_PATH}/autograd/*.yaml")
file(GLOB_RECURSE autograd_templates "${TOOLS_PATH}/autograd/templates/*")
add_custom_command(
  OUTPUT
  ${TORCH_GENERATED_CODE}
  COMMAND
  Python::Interpreter tools/setup_helpers/generate_code.py
    --native-functions-path "aten/src/ATen/native/native_functions.yaml"
    --tags-path "aten/src/ATen/native/tags.yaml"
    $<$<BOOL:${INTERN_DISABLE_AUTOGRAD}>:--disable-autograd>
    $<$<BOOL:${SELECTED_OP_LIST}>:--selected-op-list-path="${SELECTED_OP_LIST}">
    --force_schema_registration
    --gen_lazy_ts_backend
    ${GEN_PER_OPERATOR_FLAG}
  DEPENDS
    "${TORCH_ROOT}/aten/src/ATen/native/native_functions.yaml"
    "${TORCH_ROOT}/aten/src/ATen/native/tags.yaml"
    "${TORCH_ROOT}/aten/src/ATen/native/ts_native_functions.yaml"
    "${TORCH_ROOT}/torch/csrc/lazy/core/shape_inference.h"
    "${TORCH_ROOT}/torch/csrc/lazy/ts_backend/ts_native_functions.cpp"
    "${TORCH_ROOT}/aten/src/ATen/templates/DispatchKeyNativeFunctions.h"
    "${TORCH_ROOT}/aten/src/ATen/templates/DispatchKeyNativeFunctions.cpp"
    "${TORCH_ROOT}/aten/src/ATen/templates/LazyIr.h"
    "${TORCH_ROOT}/aten/src/ATen/templates/LazyNonNativeIr.h"
    "${TORCH_ROOT}/aten/src/ATen/templates/RegisterDispatchKey.cpp"
    ${autograd_python}
    ${autograd_yaml}
    ${autograd_templates}
    ${torchgen_python}
  WORKING_DIRECTORY "${TORCH_ROOT}")


# Required workaround for libtorch_python.so build
# see https://samthursfield.wordpress.com/2015/11/21/cmake-dependencies-between-targets-and-files-and-custom-commands/#custom-commands-in-different-directories
add_custom_target(
  generate-torch-sources
  DEPENDS ${TORCH_GENERATED_CODE}
  )

set(TORCH_SRCS ${GENERATED_CXX_TORCH})
list(APPEND TORCH_SRCS ${GENERATED_H_TORCH})
list(APPEND LIBTORCH_CMAKE_SRCS "")

list(APPEND LITE_EAGER_SYMOBLICATION_SRCS "")
if(USE_SOURCE_DEBUG_ON_MOBILE)
  append_filelist("libtorch_lite_eager_symbolication" LITE_EAGER_SYMOBLICATION_SRCS)
  # For source debug on lite interpreter, we have to add dependency on pickling
  # but references to read/writeArchiveAndTensor is not built for mobile
  # so this condition specifically says we are building for source debug
  # on mobile.
  if(BUILD_LITE_INTERPRETER)
    set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/serialization/pickle.cpp PROPERTIES COMPILE_FLAGS "-DC10_MOBILE -DFEATURE_TORCH_MOBILE")
  endif()
endif()

list(APPEND LITE_PROFILER_SRCS "")
if(USE_LITE_INTERPRETER_PROFILER)
  append_filelist("libtorch_edge_profiler_sources " LITE_PROFILER_SRCS)
endif()

# Switch between the full jit interpreter and lite interpreter
if(BUILD_LITE_INTERPRETER)
  append_filelist("libtorch_lite_cmake_sources" LIBTORCH_CMAKE_SRCS)
  list(APPEND LIBTORCH_CMAKE_SRCS ${LITE_EAGER_SYMOBLICATION_SRCS})
  list(APPEND LIBTORCH_CMAKE_SRCS ${LITE_PROFILER_SRCS})
  if(USE_LITE_AOTI)
    append_filelist("inductor_core_resources" LIBTORCH_CMAKE_SRCS)
  endif()
  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
else()
  append_filelist("libtorch_cmake_sources" LIBTORCH_CMAKE_SRCS)
  list(APPEND LIBTORCH_CMAKE_SRCS ${LITE_EAGER_SYMOBLICATION_SRCS})
  if(BUILD_LAZY_TS_BACKEND)
    append_filelist("lazy_tensor_ts_sources" LIBTORCH_CMAKE_SRCS)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # TODO: Delete this when https://github.com/pytorch/pytorch/issues/35026 is fixed
    set_source_files_properties(../torch/csrc/autograd/record_function_ops.cpp PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
  endif()
endif()
list(APPEND TORCH_SRCS ${LIBTORCH_CMAKE_SRCS})

if(PRINT_CMAKE_DEBUG_INFO)
  message(STATUS "Interpreter sources: ")
  foreach(tmp ${LIBTORCH_CMAKE_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()
endif()

# Mobile backend delegate srcs
if(INTERN_BUILD_MOBILE)
  set(DELEGATE_SRCS
    ${TORCH_SRC_DIR}/csrc/jit/backends/backend_debug_info.cpp
    ${TORCH_SRC_DIR}/csrc/jit/backends/backend_interface.cpp
  )
  list(APPEND TORCH_SRCS ${DELEGATE_SRCS})
  if(IOS AND USE_COREML_DELEGATE)
    set(COREML_DELEGATE_SRCS
      ${TORCH_SRC_DIR}/csrc/jit/backends/coreml/cpp/context.cpp
      ${TORCH_SRC_DIR}/csrc/jit/backends/coreml/objc/PTMCoreMLBackend.mm
      ${TORCH_SRC_DIR}/csrc/jit/backends/coreml/objc/PTMCoreMLExecutor.mm
      ${TORCH_SRC_DIR}/csrc/jit/backends/coreml/objc/PTMCoreMLCompiler.mm
      ${TORCH_SRC_DIR}/csrc/jit/backends/coreml/objc/PTMCoreMLFeatureProvider.mm
    )
    set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/backends/coreml/objc/PTMCoreMLBackend.mm PROPERTIES COMPILE_FLAGS "-fno-objc-arc")
    include_directories(${TORCH_ROOT}/third_party/nlohmann/single_include)
    list(APPEND TORCH_SRCS ${COREML_DELEGATE_SRCS})
  endif()
endif()

# Required workaround for LLVM 9 includes.
if(NOT MSVC)
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/tensorexpr/llvm_jit.cpp PROPERTIES COMPILE_FLAGS -Wno-noexcept-type)
endif()
# Disable certain warnings for GCC-9.X
if(CMAKE_COMPILER_IS_GNUCXX)
  # See https://github.com/pytorch/pytorch/issues/38856
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/tensorexpr/llvm_jit.cpp PROPERTIES COMPILE_FLAGS "-Wno-redundant-move -Wno-noexcept-type")
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/tensorexpr/llvm_codegen.cpp PROPERTIES COMPILE_FLAGS "-Wno-init-list-lifetime")
endif()

# Enable conditional FP16 arithmetic intrinsics
if(CPU_AARCH64 AND LINUX)
set_source_files_properties(${TORCH_ROOT}/aten/src/ATen/native/BlasKernel.cpp PROPERTIES COMPILE_FLAGS "-march=armv8.2-a+fp16")
endif()


if(NOT INTERN_DISABLE_MOBILE_INTERP)
  set(MOBILE_SRCS
     ${TORCH_SRC_DIR}/csrc/jit/mobile/function.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/import.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/import_data.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/interpreter.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/compatibility/model_compatibility.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/module.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/flatbuffer_loader.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/observer.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/parse_bytecode.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/parse_operators.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/quantization.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/train/export_data.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/train/optim/sgd.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/train/random.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/train/sequential.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/upgrader_mobile.cpp
     ${TORCH_SRC_DIR}/csrc/jit/serialization/flatbuffer_serializer.cpp
     )
  list(APPEND TORCH_SRCS ${MOBILE_SRCS})
  list(APPEND TORCH_SRCS ${LITE_EAGER_SYMOBLICATION_SRCS})
endif()

# This one needs to be unconditionally added as Functions.cpp is also unconditionally added
list(APPEND TORCH_SRCS
  ${TORCH_SRC_DIR}/csrc/autograd/FunctionsManual.cpp
  ${TORCH_SRC_DIR}/csrc/utils/out_types.cpp
)

if(NOT INTERN_DISABLE_AUTOGRAD AND NOT BUILD_LITE_INTERPRETER)
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/autograd/TraceTypeManual.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/VariableTypeManual.cpp
  )
endif()

if(${USE_ITT})
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/itt_wrapper.cpp
    ${TORCH_SRC_DIR}/csrc/profiler/stubs/itt.cpp
  )
endif()

if(NOT INTERN_BUILD_MOBILE AND NOT BUILD_LITE_INTERPRETER)
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/api/src/jit.cpp
    ${TORCH_SRC_DIR}/csrc/jit/mobile/compatibility/backport.cpp
    ${TORCH_SRC_DIR}/csrc/jit/mobile/compatibility/backport_manager.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/onnx.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/export.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/export_bytecode.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/export_module.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/flatbuffer_serializer.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/fuser/cpu/fused_kernel.cpp
    ${TORCH_SRC_DIR}/csrc/jit/api/module_save.cpp
    ${TORCH_SRC_DIR}/csrc/utils/byte_order.cpp
  )

  if(USE_DISTRIBUTED)
    append_filelist("libtorch_distributed_base_sources" TORCH_SRCS)
    if(NOT WIN32)
      append_filelist("libtorch_distributed_extra_sources" TORCH_SRCS)
    endif()
  endif()
endif()

if(USE_CUDA OR USE_ROCM)
  append_filelist("libtorch_cuda_core_sources" Caffe2_GPU_HIP_JIT_FUSERS_SRCS)
endif()

if(USE_CUDA)
  list(APPEND Caffe2_GPU_CU_SRCS ${Caffe2_GPU_HIP_JIT_FUSERS_SRCS})
  add_library(caffe2_nvrtc SHARED ${ATen_NVRTC_STUB_SRCS})
  if(MSVC)
    # Delay load nvcuda.dll so we can import torch compiled with cuda on a CPU-only machine
    set(DELAY_LOAD_FLAGS "-DELAYLOAD:nvcuda.dll;delayimp.lib")
  else()
    set(DELAY_LOAD_FLAGS "")
  endif()

  target_link_libraries(caffe2_nvrtc PRIVATE caffe2::nvrtc ${DELAY_LOAD_FLAGS})
  install(TARGETS caffe2_nvrtc DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  if(USE_NCCL)
    list(APPEND Caffe2_GPU_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/nccl.cpp)
  endif()
  if(USE_DISTRIBUTED)
    append_filelist("libtorch_cuda_distributed_base_sources" Caffe2_GPU_SRCS)
    if(NOT WIN32)
      append_filelist("libtorch_cuda_distributed_extra_sources" Caffe2_GPU_SRCS)
      set_source_files_properties(
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/intra_node_comm.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/CudaDMAConnectivity.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/CUDASymmetricMemory.cu
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/CUDASymmetricMemoryOps.cu
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/ProcessGroupNCCL.cpp
        PROPERTIES COMPILE_FLAGS "-DPYTORCH_C10_DRIVER_API_SUPPORTED=1"
      )
    endif()
  endif()
  set_source_files_properties(
    ${TORCH_ROOT}/aten/src/ATen/cuda/detail/LazyNVRTC.cpp
    PROPERTIES COMPILE_DEFINITIONS "NVRTC_SHORTHASH=${CUDA_NVRTC_SHORTHASH}"
  )
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/passes/frozen_conv_add_relu_fusion.cpp PROPERTIES COMPILE_FLAGS "-DUSE_CUDA=1")
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/interface.cpp PROPERTIES COMPILE_FLAGS "-DUSE_CUDA=1")
endif()

if(BUILD_ONEDNN_GRAPH)
  list(APPEND Caffe2_CPU_SRCS
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/LlgaTensorImpl.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/graph_fuser.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/graph_rewriter.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/graph_helper.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/register_interface.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/decompose_silu.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/interface.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/kernel.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/defer_size_check.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/layout_propagation.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/prepare_binary.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/guard_shape.cpp
  )
endif()

if(USE_ROCM)
  list(APPEND Caffe2_HIP_SRCS ${Caffe2_GPU_HIP_JIT_FUSERS_SRCS})
  if(USE_NCCL)
    list(APPEND Caffe2_HIP_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/nccl.cpp)
  endif()
  if(USE_DISTRIBUTED)
    append_filelist("libtorch_cuda_distributed_base_sources" Caffe2_HIP_SRCS)
    if(NOT WIN32)
      append_filelist("libtorch_cuda_distributed_extra_sources" Caffe2_HIP_SRCS)
    endif()
  endif()
  # caffe2_nvrtc's stubs to driver APIs are useful for HIP.
  # See NOTE [ ATen NVRTC Stub and HIP ]
  add_library(caffe2_nvrtc SHARED ${ATen_NVRTC_STUB_SRCS})
  target_link_libraries(caffe2_nvrtc hip::amdhip64 hiprtc::hiprtc)
  target_include_directories(caffe2_nvrtc PRIVATE ${CMAKE_BINARY_DIR})
  target_compile_definitions(caffe2_nvrtc PRIVATE USE_ROCM __HIP_PLATFORM_AMD__)
  install(TARGETS caffe2_nvrtc DESTINATION "${TORCH_INSTALL_LIB_DIR}")
endif()

if(NOT NO_API AND NOT BUILD_LITE_INTERPRETER)
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/api/src/cuda.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/datasets/mnist.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/distributed.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/random.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/sequential.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/stream.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/enum.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/imethod.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/serialize.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/jit.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/mps.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/init.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/module.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/_functions.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/activation.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/adaptive.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/batchnorm.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/normalization.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/instancenorm.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/conv.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/dropout.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/distance.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/embedding.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/fold.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/linear.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/loss.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/padding.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/pixelshuffle.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/pooling.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/rnn.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/upsampling.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/transformer.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/container/functional.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/activation.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/adaptive.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/batchnorm.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/embedding.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/instancenorm.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/normalization.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/conv.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/dropout.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/linear.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/padding.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/pooling.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/rnn.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/vision.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/transformer.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/adagrad.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/adam.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/adamw.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/lbfgs.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/optimizer.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/rmsprop.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/serialize.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/sgd.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/schedulers/lr_scheduler.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/schedulers/step_lr.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/schedulers/reduce_on_plateau_scheduler.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/serialize/input-archive.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/serialize/output-archive.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/xpu.cpp
  )
endif()

list(APPEND Caffe2_CPU_SRCS ${TORCH_SRCS})

if(USE_MPS)
  list(APPEND Caffe2_CPU_SRCS ${Caffe2_MPS_SRCS})
  if(CAN_COMPILE_METAL)
      file(TOUCH ${CMAKE_BINARY_DIR}/aten/src/ATen/metallib_dummy.cpp)
      list(APPEND Caffe2_CPU_SRCS ${CMAKE_BINARY_DIR}/aten/src/ATen/metallib_dummy.cpp)
  endif()
endif()

# NOTE [ Linking AVX and non-AVX files ]
#
# Regardless of the CPU capabilities, we build some files with AVX2, and AVX512
# instruction set. If the host CPU doesn't support those, we simply ignore their
# functions at runtime during dispatch.
#
# We must make sure that those files are at the end of the input list when
# linking the torch_cpu library. Otherwise, the following error scenario might
# occur:
# 1. A non-AVX2 and an AVX2 file both call a function defined with the `inline`
#    keyword
# 2. The compiler decides not to inline this function
# 3. Two different versions of the machine code are generated for this function:
#    one without AVX2 instructions and one with AVX2.
# 4. When linking, the AVX2 version is found earlier in the input object files,
#    so the linker makes the entire library use it, even in code not guarded by
#    the dispatcher.
# 5. A CPU without AVX2 support executes this function, encounters an AVX2
#    instruction and crashes.
#
# Thus we organize the input files in the following order:
# 1. All files with no AVX-n support
# 2. All files with AVX2 support ('*AVX2.cpp')
# 3. All files with AVX512 support ('*AVX512.cpp')
set(Caffe2_CPU_SRCS_NON_AVX)
set(Caffe2_CPU_SRCS_AVX2)
set(Caffe2_CPU_SRCS_AVX512)
foreach(input_filename ${Caffe2_CPU_SRCS})
  if(${input_filename} MATCHES "AVX2\\.cpp")
    list(APPEND Caffe2_CPU_SRCS_AVX2 ${input_filename})
  elseif(${input_filename} MATCHES "AVX512\\.cpp")
    list(APPEND Caffe2_CPU_SRCS_AVX512 ${input_filename})
  else()
    list(APPEND Caffe2_CPU_SRCS_NON_AVX ${input_filename})
  endif()
endforeach(input_filename)
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS_NON_AVX} ${Caffe2_CPU_SRCS_AVX2} ${Caffe2_CPU_SRCS_AVX512})

# ==========================================================
# END formerly-libtorch sources
# ==========================================================

if(BUILD_LIBTORCHLESS)
  find_library(TORCH_LIB torch PATHS $ENV{LIBTORCH_LIB_PATH} NO_DEFAULT_PATH)
  find_library(TORCH_CPU_LIB torch_cpu PATHS $ENV{LIBTORCH_LIB_PATH} NO_DEFAULT_PATH)

  if(USE_CUDA)
    find_library(TORCH_CUDA_LIB torch_cuda PATHS $ENV{LIBTORCH_LIB_PATH} NO_DEFAULT_PATH)
  endif()

  if(USE_ROCM)
    find_library(TORCH_HIP_LIB torch_hip PATHS $ENV{LIBTORCH_LIB_PATH} NO_DEFAULT_PATH)
  endif()

  if(USE_XPU)
    find_library(TORCH_XPU_LIB torch_xpu PATHS $ENV{LIBTORCH_LIB_PATH} NO_DEFAULT_PATH)
  endif()
  add_subdirectory(../torch torch)
  # ---[ Torch python bindings build
  set(TORCH_PYTHON_COMPILE_OPTIONS ${TORCH_PYTHON_COMPILE_OPTIONS} PARENT_SCOPE)
  set(TORCH_PYTHON_LINK_FLAGS ${TORCH_PYTHON_LINK_FLAGS} PARENT_SCOPE)
else()
  set(TORCH_LIB torch)
  set(TORCH_CPU_LIB torch_cpu)
  set(TORCH_CUDA_LIB torch_cuda)
  set(TORCH_HIP_LIB torch_hip)
  set(TORCH_XPU_LIB torch_xpu)
endif()


if(NOT BUILD_LIBTORCHLESS)
add_library(torch_cpu ${Caffe2_CPU_SRCS})
if(HAVE_SOVERSION)
  set_target_properties(torch_cpu PROPERTIES
      VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
endif()
torch_compile_options(torch_cpu)  # see cmake/public/utils.cmake

# Ignore Wdeprecated-XXX errors from third-party libraries
if(NOT MSVC)
  set_source_files_properties(${PROJECT_SOURCE_DIR}/torch/csrc/distributed/c10d/socket.cpp PROPERTIES COMPILE_OPTIONS "-Wno-error=deprecated")
endif()

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND NOT USE_IOS AND NOT USE_COREML_DELEGATE)
  target_compile_options_if_supported(torch_cpu "-Wmissing-prototypes")
  target_compile_options_if_supported(torch_cpu "-Werror=missing-prototypes")
  if(TARGET torch_cuda)
    target_compile_options_if_supported(torch_cuda "-Wmissing-prototypes")
    target_compile_options_if_supported(torch_cuda "-Werror=missing-prototypes")
  endif()
  get_target_property(TORCH_CPU_SOURCES torch_cpu SOURCES)
  foreach(generated_file IN LISTS GENERATED_CXX_TORCH)
    set_source_files_properties(${generated_file} PROPERTIES COMPILE_OPTIONS "-Wno-missing-prototypes;-Wno-error=missing-prototypes")
  endforeach()
  foreach(source_file IN LISTS TORCH_CPU_SOURCES)
    get_filename_component(source_file "${source_file}" REALPATH)
    string(FIND "${source_file}" "${CMAKE_BINARY_DIR}" res)
    if(res GREATER -1)
      set_source_files_properties(${source_file} PROPERTIES COMPILE_OPTIONS "-Wno-missing-prototypes;-Wno-error=missing-prototypes")
      continue()
    endif()
    string(FIND "${source_file}" "embedding_lookup_idx_avx2.cc" res)
    if(res GREATER -1)
      set_source_files_properties(${source_file} PROPERTIES COMPILE_OPTIONS "-Wno-missing-prototypes;-Wno-error=missing-prototypes")
    endif()
  endforeach()
endif()
if(USE_MPS)
  if(CAN_COMPILE_METAL)
    add_dependencies(torch_cpu metallibs)
    target_link_options(torch_cpu PRIVATE -Wl,-sectcreate,__TEXT,metal_basic,${CMAKE_CURRENT_BINARY_DIR}/aten/src/ATen/kernels_basic.metallib)
    target_link_options(torch_cpu PRIVATE -Wl,-sectcreate,__TEXT,metal_bfloat,${CMAKE_CURRENT_BINARY_DIR}/aten/src/ATen/kernels_bfloat.metallib)
  else()
    target_compile_definitions(torch_cpu PRIVATE PYTORCH_JIT_COMPILE_SHADERS)
  endif()
endif()

option(TORCH_USE_IWYU "Use include-what-you-use to clean up header inclusion" OFF)
if(TORCH_USE_IWYU)
  find_program(iwyu NAMES include-what-you-use)
  if(iwyu)
    set(iwyu_cmd
        "include-what-you-use"
        "-Xiwyu"
        "--transitive_includes_only"
        "-Xiwyu"
        "--no_fwd_decls"
        "-Xiwyu"
        "--prefix_header_includes=keep"
        "-Xiwyu"
        "--mapping_file=${CMAKE_CURRENT_LIST_DIR}/../tools/iwyu/all.imp"
        )
    set_property(TARGET torch_cpu PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${iwyu_cmd})
  endif()
endif()

set_property(SOURCE ${ATen_CORE_SRCS} APPEND
    PROPERTY COMPILE_DEFINITIONS "TORCH_ASSERT_ONLY_METHOD_OPERATORS")
set_property(SOURCE ${ATen_ATTENTION_KERNEL_SRCS} APPEND
    PROPERTY COMPILE_DEFINITIONS "TORCH_ASSERT_NO_OPERATORS")

if(USE_MPS OR USE_PYTORCH_METAL)
  enable_language(OBJC OBJCXX)
endif()

if(USE_PRECOMPILED_HEADERS)
  target_precompile_headers(torch_cpu PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:ATen/core/ATen_pch.h>")
  # Exclude some files from using PCH
  set_source_files_properties(
      # Not built with OpenMP, so PCH is invalid
      ${Torch_SOURCE_DIR}/aten/src/ATen/MapAllocator.cpp
      # Builds with incompatible compiler flags
      ${Caffe2_CPU_SRCS_AVX2}
      ${Caffe2_CPU_SRCS_AVX512}
      PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
endif()

# Pass path to PocketFFT
if(AT_POCKETFFT_ENABLED)
  set_source_files_properties(
      "${PROJECT_SOURCE_DIR}/aten/src/ATen/native/mkl/SpectralOps.cpp"
      PROPERTIES INCLUDE_DIRECTORIES "${POCKETFFT_INCLUDE_DIR}")
endif()

if(CMAKE_COMPILER_IS_GNUCXX AND BUILD_LIBTORCH_CPU_WITH_DEBUG)
  # To enable debug fission we need to build libtorch_cpu with debug info on,
  # but this increases link time and peak memory usage if we use the
  # REL_WITH_DEB_INFO env var since that enables it for everything, but it's
  # only really necessary for libtorch_cpu.
  target_compile_options(torch_cpu PRIVATE "-g")
endif()

if(USE_LLVM AND LLVM_FOUND)
  llvm_map_components_to_libnames(LLVM_LINK_LIBS
    support core analysis executionengine instcombine
    scalaropts transformutils ${LLVM_TARGETS_TO_BUILD} orcjit)
  target_link_libraries(torch_cpu PRIVATE ${LLVM_LINK_LIBS})
  if(APPLE)
    set(LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/unexported_symbols.lds")
    set_target_properties(torch_cpu PROPERTIES LINK_DEPENDS ${LINKER_SCRIPT})
    set_target_properties(torch_cpu PROPERTIES LINK_FLAGS "-Wl,-unexported_symbols_list,${LINKER_SCRIPT}")
  elseif(UNIX)
    set(LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/version_script.lds")
    set_target_properties(torch_cpu PROPERTIES LINK_DEPENDS ${LINKER_SCRIPT})
    target_link_libraries(torch_cpu PRIVATE "-Wl,--version-script=${LINKER_SCRIPT}")
  endif()
endif(USE_LLVM AND LLVM_FOUND)

# This is required for older versions of CMake, which don't allow
# specifying add_library() without a list of source files
set(DUMMY_EMPTY_FILE ${CMAKE_BINARY_DIR}/empty.cpp)

if(MSVC)
  set(DUMMY_FILE_CONTENT "__declspec(dllexport) int ignore_this_library_placeholder(){return 0\\;}")
else()
  set(DUMMY_FILE_CONTENT "")
endif()

file(WRITE ${DUMMY_EMPTY_FILE} ${DUMMY_FILE_CONTENT})

# Wrapper library for people who link against torch and expect both CPU and CUDA support
# Contains "torch_cpu" and "torch_cuda"
add_library(torch ${DUMMY_EMPTY_FILE})
if(HAVE_SOVERSION)
  set_target_properties(torch PROPERTIES
      VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
endif()

if(USE_ROCM)
  filter_list(__caffe2_hip_srcs_cpp Caffe2_HIP_SRCS "\\.(cu|hip)$")
  set_source_files_properties(${__caffe2_hip_srcs_cpp} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
endif()

# Compile exposed libraries.
if(USE_ROCM)
  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
  list(APPEND Caffe2_HIP_SRCS ${GENERATED_CXX_TORCH_CUDA})
  hip_add_library(torch_hip ${Caffe2_HIP_SRCS})
  if(USE_FLASH_ATTENTION)
    target_link_libraries(torch_hip PRIVATE __caffe2_aotriton)
  endif()
  set(CUDA_LINK_LIBRARIES_KEYWORD)
  torch_compile_options(torch_hip)  # see cmake/public/utils.cmake
  # TODO: Not totally sure if this is live or not
  if(USE_NCCL)
    target_link_libraries(torch_hip PRIVATE __caffe2_nccl)
    target_compile_definitions(torch_hip PRIVATE USE_NCCL)
  endif()

  if(USE_PRECOMPILED_HEADERS)
    target_precompile_headers(torch_hip PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:ATen/core/ATen_pch.h>")
  endif()
elseif(USE_CUDA)
  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
  list(APPEND Caffe2_GPU_SRCS ${GENERATED_CXX_TORCH_CUDA})
  if(CUDA_SEPARABLE_COMPILATION)
    # Separate compilation fails when kernels using `thrust::sort_by_key`
    # are linked with the rest of CUDA code. Workaround by linking them separately.
    add_library(torch_cuda ${Caffe2_GPU_SRCS} ${Caffe2_GPU_CU_SRCS})
    set_property(TARGET torch_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    add_library(torch_cuda_w_sort_by_key OBJECT
        ${Caffe2_GPU_SRCS_W_SORT_BY_KEY}
        ${Caffe2_GPU_CU_SRCS_W_SORT_BY_KEY})
    set_property(TARGET torch_cuda_w_sort_by_key PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
    target_link_libraries(torch_cuda PRIVATE torch_cuda_w_sort_by_key)
  else()
    add_library(torch_cuda
        ${Caffe2_GPU_SRCS} ${Caffe2_GPU_SRCS_W_SORT_BY_KEY}
        ${Caffe2_GPU_CU_SRCS} ${Caffe2_GPU_CU_SRCS_W_SORT_BY_KEY})
  endif()
  set(CUDA_LINK_LIBRARIES_KEYWORD)
  torch_compile_options(torch_cuda)  # see cmake/public/utils.cmake
  target_compile_definitions(torch_cuda PRIVATE USE_CUDA)

  if(USE_CUFILE)
    target_link_libraries(torch_cuda PRIVATE torch::cufile)
    target_compile_definitions(torch_cuda PRIVATE USE_CUFILE)
  endif()
  if(USE_CUSPARSELT)
      target_link_libraries(torch_cuda PRIVATE torch::cusparselt)
      target_compile_definitions(torch_cuda PRIVATE USE_CUSPARSELT)
  endif()
  if(USE_CUDSS)
    target_link_libraries(torch_cuda PRIVATE torch::cudss)
    target_compile_definitions(torch_cuda PRIVATE USE_CUDSS)
  endif()
  if(USE_NCCL)
    target_link_libraries(torch_cuda PRIVATE __caffe2_nccl)
    target_compile_definitions(torch_cuda PRIVATE USE_NCCL)
  endif()
  if(USE_UCC)
    target_link_libraries(torch_cuda PRIVATE __caffe2_ucc)
    target_compile_definitions(torch_cuda PRIVATE USE_UCC)
  endif()
  if(USE_FLASH_ATTENTION)
    target_compile_definitions(torch_cuda PRIVATE USE_FLASH_ATTENTION)
  endif()
  if(USE_MEM_EFF_ATTENTION)
    target_compile_definitions(torch_cuda PRIVATE USE_MEM_EFF_ATTENTION)
  endif()
  if(BUILD_LAZY_CUDA_LINALG)
    add_library(torch_cuda_linalg ${ATen_CUDA_LINALG_SRCS})
    target_compile_definitions(torch_cuda_linalg PRIVATE USE_CUDA BUILD_LAZY_CUDA_LINALG)
    # Library order is important during static linking
    # `torch::magma` should be mentioned before other CUDA
    # to transitively include all symbols present in torch_cuda/torch_cpu
    if(USE_MAGMA)
      target_link_libraries(torch_cuda_linalg PRIVATE torch::magma)
      # CUDAHooks reports version of MAGMA PyTorch was compiled against, i.e. needs to be able to include magma headers
      get_target_property(HOOKS_INCLUDE_DIRECTORIES torch_cuda INCLUDE_DIRECTORIES)
      if(NOT "${MAGMA_INCLUDE_DIR}" IN_LIST HOOKS_INCLUDE_DIRECTORIES)
        set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/cuda/detail/CUDAHooks.cpp PROPERTIES INCLUDE_DIRECTORIES  "${MAGMA_INCLUDE_DIR}")
      endif()
    endif()
    target_link_libraries(torch_cuda_linalg PRIVATE
        torch_cpu
        torch_cuda
    )
    if($ENV{ATEN_STATIC_CUDA})
      if(CUDA_VERSION_MAJOR LESS_EQUAL 11)
        target_link_libraries(torch_cuda_linalg PRIVATE
            CUDA::cusolver_static
            ${CUDAToolkit_LIBRARY_DIR}/liblapack_static.a     # needed for libcusolver_static
        )
      elseif(CUDA_VERSION_MAJOR GREATER_EQUAL 12)
        target_link_libraries(torch_cuda_linalg PRIVATE
            CUDA::cusolver_static
            ${CUDAToolkit_LIBRARY_DIR}/libcusolver_lapack_static.a     # needed for libcusolver_static
        )
      endif()
    else()
      target_link_libraries(torch_cuda_linalg PRIVATE
          CUDA::cusolver
      )
    endif()
    # NS: TODO, is this really necessary?
    if(USE_MAGMA AND CAFFE2_STATIC_LINK_CUDA)
      target_link_libraries(torch_cuda_linalg PRIVATE
          CUDA::culibos ${CMAKE_DL_LIBS})
    endif()
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/native/cuda/LinearAlgebraStubs.cpp PROPERTIES COMPILE_FLAGS "-DBUILD_LAZY_CUDA_LINALG")
    install(TARGETS torch_cuda_linalg DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  endif()

  if(USE_PRECOMPILED_HEADERS)
    target_precompile_headers(torch_cuda PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:ATen/core/ATen_pch.h>")
  endif()

  # Apply suggestion from comment https://github.com/pytorch/pytorch/issues/113053#issuecomment-2115375714
  if(LINUX)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/cuda/CUDASparseDescriptors.cpp PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/cuda/CUDASparseBlas.cpp PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/native/sparse/cuda/SparseCUDABlas.cpp PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
  endif()
endif()

if(USE_XPU)
  add_library(torch_xpu ${Caffe2_XPU_SRCS})
  torch_compile_options(torch_xpu)  # see cmake/public/utils.cmake
  target_compile_definitions(torch_xpu PRIVATE USE_XPU)
  if(WIN32)
    target_compile_options(torch_xpu PRIVATE /permissive-)
  endif()

  # ATen XPU implementation
  set(TORCH_XPU_OPS_DIR ${TORCH_ROOT}/third_party/torch-xpu-ops)
  set(TORCH_XPU_OPS_REPO_URL https://github.com/intel/torch-xpu-ops.git)
  file(READ "${TORCH_ROOT}/third_party/xpu.txt" TORCH_XPU_OPS_COMMIT)
  string(REGEX REPLACE "\n$" "" TORCH_XPU_OPS_COMMIT "${TORCH_XPU_OPS_COMMIT}")
  if(NOT EXISTS "${TORCH_XPU_OPS_DIR}/.git")
    execute_process(
      COMMAND git clone --quiet ${TORCH_XPU_OPS_REPO_URL} ${TORCH_XPU_OPS_DIR}
      RESULT_VARIABLE _exitcode)
    if(NOT _exitcode EQUAL 0)
      message(FATAL_ERROR "Fail to clone ${TORCH_XPU_OPS_REPO_URL}")
    endif()
  endif()
  execute_process(
    COMMAND git fetch --quiet
    WORKING_DIRECTORY ${TORCH_XPU_OPS_DIR}
    RESULT_VARIABLE _exitcode)
  if(NOT _exitcode EQUAL 0)
    message(FATAL_ERROR "Fail to fetch ${TORCH_XPU_OPS_REPO_URL}")
  endif()
  execute_process(
    COMMAND git checkout --quiet ${TORCH_XPU_OPS_COMMIT}
    WORKING_DIRECTORY ${TORCH_XPU_OPS_DIR}
    RESULT_VARIABLE _exitcode)
  if(NOT _exitcode EQUAL 0)
    message(FATAL_ERROR "Fail to checkout ${TORCH_XPU_OPS_REPO_URL} to ${TORCH_XPU_OPS_COMMIT}")
  endif()

  set(TORCH_XPU_OPS_INCLUDE_DIRS
      ${TORCH_SRC_DIR}/csrc/api
      ${TORCH_SRC_DIR}/csrc/api/include
      ${Caffe2_CPU_INCLUDE}
      ${Caffe2_XPU_INCLUDE})
  # Pass the target as a dependency so that ATen headers generation
  # could be followed by torch-xpu-ops build.
  # 1. Sources in torch-xpu-ops depend on generated ATen headers.
  # 2. Using add_custom_command in torch-xpu-ops to define sycl device sources
  #    compilation. add_custom_command requires an explicit dependency.
  list(APPEND ${Caffe2_XPU_INCLUDE} ${TORCH_XPU_OPS_DIR}/src/ATen/)
  set(TORCH_XPU_OPS_PYTORCH_DEPS ATEN_CPU_FILES_GEN_TARGET)

  add_subdirectory(${TORCH_ROOT}/third_party/torch-xpu-ops
      ${CMAKE_BINARY_DIR}/caffe2/aten_xpu)
  if(NOT TARGET torch_xpu_ops)
    message(WARNING "Failed to include ATen XPU implementation target")
  else()
    target_link_libraries(torch_xpu PRIVATE torch_xpu_ops)
    if(MSVC)
      # Windows
      target_link_options(torch_xpu PRIVATE
      "-WHOLEARCHIVE:$<TARGET_FILE:torch_xpu_ops>")
    else()
      # Linux
      target_link_options(torch_xpu PRIVATE
          "-Wl,--whole-archive,$<TARGET_FILE:torch_xpu_ops>,--no-whole-archive")
    endif()

    # Set cached ${ATen_XPU_INCLUDE_DIRS} to torch
    include_directories(SYSTEM ${ATen_XPU_INCLUDE_DIRS})

  endif()
endif()

if(NOT MSVC AND USE_XNNPACK)
  TARGET_LINK_LIBRARIES(torch_cpu PRIVATE fxdiv)
endif()

# ==========================================================
# formerly-libtorch flags
# ==========================================================


# Build model tracer for tracing-based selective build
if(TRACING_BASED AND NOT BUILD_LITE_INTERPRETER AND NOT INTERN_BUILD_MOBILE)
  add_subdirectory(
    ${TORCH_ROOT}/torch/csrc/jit/mobile/model_tracer
    ${CMAKE_BINARY_DIR}/model_tracer
  )
  string(APPEND CMAKE_CXX_FLAGS " -DENABLE_RECORD_KERNEL_FUNCTION_DTYPE")
endif()

# Codegen selected_mobile_ops.h for template selective build
if(BUILD_LITE_INTERPRETER AND SELECTED_OP_LIST)
  message("running gen_selected_mobile_ops_header for:  '${SELECTED_OP_LIST}'")
  file(GLOB lite_interpreter_python "${TOOLS_PATH}/lite_interpreter/*.py")
  if(${TRACING_BASED})
    file(GLOB code_analyzer_python "${TOOLS_PATH}/code_analyzer/*.py")
    add_custom_command(
      OUTPUT ${CMAKE_BINARY_DIR}/aten/src/ATen/selected_mobile_ops.h
      COMMAND
        Python::Interpreter
        -m tools.code_analyzer.gen_oplist
        --model_file_list_path "${SELECTED_OP_LIST}"
        --output_dir "${CMAKE_BINARY_DIR}/aten/src/ATen"
      DEPENDS
        ${torchgen_python}
        ${lite_interpreter_python}
        ${code_analyzer_python}
        "${SELECTED_OP_LIST}"
        "${TORCH_ROOT}/aten/src/ATen/native/native_functions.yaml"
      WORKING_DIRECTORY "${TORCH_ROOT}")
  else()
    add_custom_command(
      OUTPUT ${CMAKE_BINARY_DIR}/aten/src/ATen/selected_mobile_ops.h
      COMMAND
        Python::Interpreter
        -m tools.lite_interpreter.gen_selected_mobile_ops_header
        --yaml_file_path "${SELECTED_OP_LIST}"
        --output_file_path "${CMAKE_BINARY_DIR}/aten/src/ATen"
      DEPENDS
        ${torchgen_python}
        ${lite_interpreter_python}
        "${SELECTED_OP_LIST}"
        "${TORCH_ROOT}/aten/src/ATen/native/native_functions.yaml"
      WORKING_DIRECTORY "${TORCH_ROOT}")
  endif()

  add_custom_target(
    __selected_mobile_ops_header_gen
    DEPENDS ${CMAKE_BINARY_DIR}/aten/src/ATen/selected_mobile_ops.h)
  add_dependencies(torch_cpu __selected_mobile_ops_header_gen)
endif()

if(NOT NO_API)
  target_include_directories(torch_cpu PRIVATE
    ${TORCH_SRC_DIR}/csrc/api
    ${TORCH_SRC_DIR}/csrc/api/include)
endif()

if(USE_CUDA AND MSVC)
  # -INCLUDE is used to ensure torch_cuda is linked against in a project that relies on them.
  # Related issue: https://github.com/pytorch/pytorch/issues/31611
  target_link_libraries(torch_cuda INTERFACE "-INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

if(NOT BUILD_LITE_INTERPRETER)
  set(TH_CPU_INCLUDE
    # dense
    aten/src/TH
    ${CMAKE_CURRENT_BINARY_DIR}/aten/src/TH
    ${TORCH_ROOT}/aten/src
    ${CMAKE_CURRENT_BINARY_DIR}/aten/src

    ${CMAKE_BINARY_DIR}/aten/src)
    target_include_directories(torch_cpu PRIVATE ${TH_CPU_INCLUDE})
endif()

set(ATen_CPU_INCLUDE
  ${TORCH_ROOT}/aten/src
  ${CMAKE_CURRENT_BINARY_DIR}/../aten/src
  ${CMAKE_BINARY_DIR}/aten/src)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/native/QuantizedLinear.cpp PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/native/RNN.cpp PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/native/quantized/cpu/qlinear_prepack.cpp PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/native/quantized/qlinear_unpack.cpp PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
endif()

target_include_directories(torch_cpu PRIVATE ${ATen_CPU_INCLUDE})

target_include_directories(torch_cpu PRIVATE
  ${TORCH_SRC_DIR}/csrc)

target_include_directories(torch_cpu PRIVATE
  ${TORCH_ROOT}/third_party/miniz-2.1.0)

target_include_directories(torch_cpu PRIVATE
  ${TORCH_ROOT}/third_party/kineto/libkineto/include)

if(USE_KINETO)
  target_include_directories(torch_cpu PRIVATE
    ${TORCH_ROOT}/third_party/kineto/libkineto/src)
endif()

target_include_directories(torch_cpu PRIVATE
  ${TORCH_ROOT}/third_party/cpp-httplib)

target_include_directories(torch_cpu PRIVATE
  ${TORCH_ROOT}/third_party/nlohmann/include)

install(DIRECTORY "${TORCH_SRC_DIR}/csrc"
  DESTINATION ${TORCH_INSTALL_INCLUDE_DIR}/torch
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
install(FILES
  "${TORCH_SRC_DIR}/script.h"
  "${TORCH_SRC_DIR}/extension.h"
  "${TORCH_SRC_DIR}/custom_class.h"
  "${TORCH_SRC_DIR}/library.h"
  "${TORCH_SRC_DIR}/custom_class_detail.h"
  DESTINATION ${TORCH_INSTALL_INCLUDE_DIR}/torch)
if(BUILD_TEST)
  if(BUILD_EXECUTORCH)
    add_subdirectory(
            ${TORCH_ROOT}/test/edge
            ${CMAKE_BINARY_DIR}/test_edge_op_registration
    )
  endif()
  if(BUILD_LITE_INTERPRETER)
    add_subdirectory(
      ${TORCH_ROOT}/test/cpp/lite_interpreter_runtime
      ${CMAKE_BINARY_DIR}/test_lite_interpreter_runtime
    )
    add_subdirectory(
      ${TORCH_ROOT}/test/mobile/lightweight_dispatch
      ${CMAKE_BINARY_DIR}/test_codegen_unboxing
    )
  else()
    add_subdirectory(${TORCH_ROOT}/test/cpp/jit ${CMAKE_BINARY_DIR}/test_jit)
    add_subdirectory(${TORCH_ROOT}/test/inductor ${CMAKE_BINARY_DIR}/test_inductor)
    add_subdirectory(
      ${TORCH_ROOT}/test/cpp/tensorexpr
      ${CMAKE_BINARY_DIR}/test_tensorexpr
    )
    if(USE_DISTRIBUTED)
      add_subdirectory(${TORCH_ROOT}/test/cpp/c10d ${CMAKE_BINARY_DIR}/test_cpp_c10d)
      if(NOT WIN32)
        add_subdirectory(${TORCH_ROOT}/test/cpp/dist_autograd ${CMAKE_BINARY_DIR}/dist_autograd)
        add_subdirectory(${TORCH_ROOT}/test/cpp/rpc ${CMAKE_BINARY_DIR}/test_cpp_rpc)
      endif()
    endif()
    if(NOT NO_API)
      add_subdirectory(${TORCH_ROOT}/test/cpp/api ${CMAKE_BINARY_DIR}/test_api)
    endif()

    if(USE_LLVM AND LLVM_FOUND)
      add_subdirectory(
        ${TORCH_ROOT}/test/mobile/nnc
        ${CMAKE_BINARY_DIR}/test_mobile_nnc
      )
    endif()
    add_subdirectory(${TORCH_ROOT}/test/cpp/lazy
                     ${CMAKE_BINARY_DIR}/test_lazy)
  endif()
  if(BUILD_AOT_INDUCTOR_TEST)
    add_subdirectory(
      ${TORCH_ROOT}/test/cpp/aoti_abi_check
      ${CMAKE_BINARY_DIR}/test_aoti_abi_check)
    add_subdirectory(
      ${TORCH_ROOT}/test/cpp/aoti_inference
      ${CMAKE_BINARY_DIR}/test_aoti_inference)
  endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  include(../cmake/CheckAbi.cmake)
endif()

# CMake config for external projects.
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/TorchConfigVersion.cmake.in
  ${PROJECT_BINARY_DIR}/TorchConfigVersion.cmake
  @ONLY)
configure_file(
  ${TORCH_ROOT}/cmake/TorchConfig.cmake.in
  ${PROJECT_BINARY_DIR}/TorchConfig.cmake
  @ONLY)
install(FILES
  ${PROJECT_BINARY_DIR}/TorchConfigVersion.cmake
  ${PROJECT_BINARY_DIR}/TorchConfig.cmake
  DESTINATION share/cmake/Torch)

# ---[ Torch python bindings build
add_subdirectory(../torch torch)
set(TORCH_PYTHON_COMPILE_OPTIONS ${TORCH_PYTHON_COMPILE_OPTIONS} PARENT_SCOPE)
set(TORCH_PYTHON_LINK_FLAGS ${TORCH_PYTHON_LINK_FLAGS} PARENT_SCOPE)
# ==========================================================
# END formerly-libtorch flags
# ==========================================================

if(NOT NO_API)
  target_include_directories(torch_cpu PUBLIC
    $<BUILD_INTERFACE:${TORCH_SRC_DIR}/csrc/api>
    $<BUILD_INTERFACE:${TORCH_SRC_DIR}/csrc/api/include>)
endif()

if(USE_ROCM)
  target_compile_definitions(torch_hip PRIVATE
    USE_ROCM
    __HIP_PLATFORM_AMD__
    )
  # NB: Massive hack.  torch/csrc/jit/codegen/fuser/codegen.cpp includes
  # torch/csrc/jit/codegen/fuser/cuda/resource_strings.h which changes the
  # strings depending on if you're __HIP_PLATFORM_AMD__ or not.
  # But that file is in torch_cpu!  So, against all odds, this macro
  # has to be set on torch_cpu too.  I also added it to torch for
  # better luck
  target_compile_definitions(torch_cpu PRIVATE
    USE_ROCM
    __HIP_PLATFORM_AMD__
    )
  target_compile_definitions(torch PRIVATE
    USE_ROCM
    __HIP_PLATFORM_AMD__
    )

  if(NOT ROCM_SOURCE_DIR)
    set(ROCM_SOURCE_DIR "$ENV{ROCM_SOURCE_DIR}")
  endif()
  if($ROCM_SOURCE_DIR STREQUAL "")
    set(ROCM_SOURCE_DIR "/opt/rocm")
  endif()
  message(INFO "caffe2 ROCM_SOURCE_DIR = ${ROCM_SOURCE_DIR}")
  target_include_directories(torch_hip PRIVATE
    ${ROCM_SOURCE_DIR}/include
    ${ROCM_SOURCE_DIR}/hcc/include
    ${ROCM_SOURCE_DIR}/rocblas/include
    ${ROCM_SOURCE_DIR}/hipsparse/include
    ${ROCM_SOURCE_DIR}/include/rccl/
    )
  if(USE_FLASH_ATTENTION)
    target_compile_definitions(torch_hip PRIVATE USE_FLASH_ATTENTION)
  endif()
  if(USE_MEM_EFF_ATTENTION)
    target_compile_definitions(torch_hip PRIVATE USE_MEM_EFF_ATTENTION)
  endif()
endif()

if(BUILD_LITE_INTERPRETER)
  target_compile_definitions(torch_cpu PRIVATE BUILD_LITE_INTERPRETER)
  # Enable template selective build only when SELECTED_OP_LIST is provided.
  if(SELECTED_OP_LIST)
    target_compile_definitions(torch_cpu PRIVATE TEMPLATE_SELECTIVE_BUILD)
  endif()
endif()


# Pass USE_DISTRIBUTED to torch_cpu, as some codes in jit/pickler.cpp and
# jit/unpickler.cpp need to be compiled only when USE_DISTRIBUTED is set
if(USE_DISTRIBUTED)
  target_compile_definitions(torch_cpu PUBLIC USE_DISTRIBUTED)
  if(USE_GLOO AND USE_C10D_GLOO)
    target_compile_definitions(torch_cpu PUBLIC USE_C10D_GLOO)
  endif()
  if(USE_UCC AND USE_C10D_UCC)
    target_compile_definitions(torch_cpu PUBLIC USE_C10D_UCC)
    if(USE_CUDA)
      target_compile_definitions(torch_cuda PUBLIC USE_C10D_UCC)
    endif()
  endif()
  if(USE_NCCL AND USE_C10D_NCCL)
    if(USE_ROCM)
      target_compile_definitions(torch_hip PUBLIC USE_C10D_NCCL)
    else()
      target_compile_definitions(torch_cuda PUBLIC USE_C10D_NCCL)
    endif()
  endif()
  if(USE_MPI AND USE_C10D_MPI)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set_source_files_properties(
        "${TORCH_SRC_DIR}/csrc/distributed/c10d/ProcessGroupMPI.cpp"
        PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
    endif()
    target_compile_definitions(torch_cpu PUBLIC USE_C10D_MPI)
  endif()
  # Pass USE_RPC in order to reduce use of
  # #if defined(USE_DISTRIBUTED) && !defined(_WIN32)
  # need to be removed when RPC is supported
  if(NOT WIN32)
    target_compile_definitions(torch_cpu PUBLIC USE_RPC)
  endif()
  # Pass USE_TENSORPIPE to torch_cpu as some parts of rpc/utils.cpp
  # can only be compiled with USE_TENSORPIPE is set.
  if(USE_TENSORPIPE)
    target_compile_definitions(torch_cpu PUBLIC USE_TENSORPIPE)
  endif()
endif()

if(NOT INTERN_BUILD_MOBILE)
  if(${CAFFE2_LINK_LOCAL_PROTOBUF})
    target_link_libraries(torch_cpu INTERFACE protobuf::libprotobuf)
  else()
    target_link_libraries(torch_cpu PUBLIC protobuf::libprotobuf)
  endif()
endif()

if($ENV{TH_BINARY_BUILD})
  if(NOT MSVC AND USE_CUDA AND NOT APPLE)
    # Note [Extra MKL symbols for MAGMA in torch_cpu]
    #
    # When we build CUDA libraries and link against MAGMA, MAGMA makes use of
    # some BLAS symbols in its CPU fallbacks when it has no GPU versions
    # of kernels.  Previously, we ensured the BLAS symbols were filled in by
    # MKL by linking torch_cuda with BLAS, but when we are statically linking
    # against MKL (when we do wheel builds), this actually ends up pulling in a
    # decent chunk of MKL into torch_cuda, inflating our torch_cuda binary
    # size by 8M.  torch_cpu exposes most of the MKL symbols we need, but
    # empirically we determined that there are four which it doesn't provide.  If
    # we link torch_cpu with these --undefined symbols, we can ensure they
    # do get pulled in, and then we can avoid statically linking in MKL to
    # torch_cuda at all!
    #
    # We aren't really optimizing for binary size on Windows (and this link
    # line doesn't work on Windows), so don't do it there.
    #
    # These linker commands do not work on OS X, do not attempt this there.
    # (It shouldn't matter anyway, though, because OS X has dropped CUDA support)
    foreach(_symb  slaed0 daled0 dormql sormql zheevd cheevd)
    STRING(APPEND _undefined_link_flags " -Wl,--undefined=mkl_lapack_${_symb}")
    endforeach(_symb)
    set_target_properties(torch_cpu PROPERTIES LINK_FLAGS  ${_undefined_link_flags})

  endif()
endif()

target_link_libraries(torch_cpu PUBLIC c10)
target_link_libraries(torch_cpu PUBLIC ${Caffe2_PUBLIC_DEPENDENCY_LIBS})
target_link_libraries(torch_cpu PRIVATE ${Caffe2_DEPENDENCY_LIBS})
target_link_libraries(torch_cpu PRIVATE ${Caffe2_DEPENDENCY_WHOLE_LINK_LIBS})
if(USE_MPI)
  target_link_libraries(torch_cpu PRIVATE MPI::MPI_CXX)
endif()
target_include_directories(torch_cpu INTERFACE $<INSTALL_INTERFACE:include>)
target_include_directories(torch_cpu PRIVATE ${Caffe2_CPU_INCLUDE})
target_include_directories(torch_cpu SYSTEM PRIVATE "${Caffe2_DEPENDENCY_INCLUDE}")

target_compile_definitions(torch_cpu PRIVATE CAFFE2_BUILD_MAIN_LIB)
if(USE_CUDA)
  target_compile_definitions(torch_cuda PRIVATE TORCH_CUDA_BUILD_MAIN_LIB)
elseif(USE_ROCM)
  target_compile_definitions(torch_hip PRIVATE TORCH_HIP_BUILD_MAIN_LIB)
endif()

if(USE_XPU)
  target_compile_definitions(torch_xpu PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
endif()

set(EXPERIMENTAL_SINGLE_THREAD_POOL "0" CACHE STRING
  "Experimental option to use a single thread pool for inter- and intra-op parallelism")
if("${EXPERIMENTAL_SINGLE_THREAD_POOL}")
  target_compile_definitions(torch_cpu PUBLIC "-DAT_EXPERIMENTAL_SINGLE_THREAD_POOL=1")
endif()

if(MSVC AND BUILD_SHARED_LIBS)
  # ONNX is linked statically and needs to be exported from this library
  # to be used externally. Make sure that references match the export.
  target_compile_options(torch_cpu PRIVATE "-DONNX_BUILD_MAIN_LIB")
endif()

caffe2_interface_library(torch_cpu torch_cpu_library)

if(USE_CUDA)
  caffe2_interface_library(torch_cuda torch_cuda_library)
elseif(USE_ROCM)
  caffe2_interface_library(torch_hip torch_hip_library)
elseif(USE_XPU)
  caffe2_interface_library(torch_xpu torch_xpu_library)
endif()

caffe2_interface_library(torch torch_library)

install(TARGETS torch_cpu torch_cpu_library EXPORT Caffe2Targets DESTINATION "${TORCH_INSTALL_LIB_DIR}")

if(USE_CUDA)
  install(TARGETS torch_cuda torch_cuda_library EXPORT Caffe2Targets DESTINATION "${TORCH_INSTALL_LIB_DIR}")
elseif(USE_ROCM)
  install(TARGETS torch_hip torch_hip_library EXPORT Caffe2Targets DESTINATION "${TORCH_INSTALL_LIB_DIR}")
elseif(USE_XPU)
  install(TARGETS torch_xpu torch_xpu_library EXPORT Caffe2Targets DESTINATION "${TORCH_INSTALL_LIB_DIR}")
endif()

install(TARGETS torch torch_library EXPORT Caffe2Targets DESTINATION "${TORCH_INSTALL_LIB_DIR}")

target_link_libraries(torch PUBLIC torch_cpu_library)

if(USE_CUDA)
  target_link_libraries(torch PUBLIC torch_cuda_library)
elseif(USE_ROCM)
  target_link_libraries(torch PUBLIC torch_hip_library)
endif()

if(USE_XPU)
  target_link_libraries(torch PUBLIC torch_xpu_library)
endif()

if(PRINT_CMAKE_DEBUG_INFO)
  print_target_properties(torch)
  print_target_properties(torch_cpu)
endif()

# Install PDB files for MSVC builds
if(MSVC AND BUILD_SHARED_LIBS)
  install(FILES $<TARGET_PDB_FILE:torch_cpu> DESTINATION "${TORCH_INSTALL_LIB_DIR}" OPTIONAL)
  if(USE_CUDA)
    install(FILES $<TARGET_PDB_FILE:torch_cuda> DESTINATION "${TORCH_INSTALL_LIB_DIR}" OPTIONAL)
  elseif(USE_ROCM)
    install(FILES $<TARGET_PDB_FILE:torch_hip> DESTINATION "${TORCH_INSTALL_LIB_DIR}" OPTIONAL)
  endif()
endif()

# ---[ CUDA library.
if(USE_CUDA)
  # FIXME: If kineto is linked with CUPTI it pollutes torch_cpu with CUDA dependencies
  # Even worse, it never declares that it depends on cudart, but calls the API, see
  # https://github.com/pytorch/kineto/blob/aef2f5c0f15e3be52406ac0b885e8689de6bc9f6/libkineto/src/CudaDeviceProperties.cpp#L24
  if(USE_KINETO AND NOT MSVC AND NOT LIBKINETO_NOCUPTI)
    target_link_libraries(torch_cpu PRIVATE torch::cudart)
  endif()
  target_link_libraries(torch_cuda INTERFACE torch::cudart)
  target_link_libraries(torch_cuda PUBLIC c10_cuda)
  if(TARGET torch::nvtx3)
    target_link_libraries(torch_cuda PRIVATE torch::nvtx3)
  else()
    target_link_libraries(torch_cuda PUBLIC torch::nvtoolsext)
  endif()

  target_include_directories(
      torch_cuda INTERFACE $<INSTALL_INTERFACE:include>)
  target_include_directories(
      torch_cuda PRIVATE ${Caffe2_GPU_INCLUDE})
  target_link_libraries(
      torch_cuda PRIVATE ${Caffe2_CUDA_DEPENDENCY_LIBS})

  # These public dependencies must go after the previous dependencies, as the
  # order of the libraries in the linker call matters here when statically
  # linking; libculibos and cublas must be last.
  target_link_libraries(torch_cuda PUBLIC torch_cpu_library ${Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS})
endif()

# ---[ XPU library.
if(USE_XPU)
  target_link_libraries(torch_xpu INTERFACE torch::xpurt)

  target_link_libraries(torch_xpu PUBLIC c10_xpu)

  target_include_directories(
      torch_xpu INTERFACE $<INSTALL_INTERFACE:include>)
  target_include_directories(
      torch_xpu PRIVATE ${Caffe2_XPU_INCLUDE})
  target_link_libraries(
      torch_xpu PRIVATE ${Caffe2_XPU_DEPENDENCY_LIBS})

  # Ensure that torch_cpu is ready before being linked by torch_xpu.
  add_dependencies(torch_xpu torch_cpu)

  if(MSVC)
    target_link_libraries(torch_xpu PUBLIC torch_cpu_library)
  else()
    include(CheckLinkerFlag)

    # Check whether the compiler supports '--no-as-needed' and '--as-needed'
    check_linker_flag(CXX "-Wl,--no-as-needed" HAVE_NO_AS_NEEDED)
    check_linker_flag(CXX "-Wl,--as-needed" HAVE_AS_NEEDED)

    if(HAVE_NO_AS_NEEDED AND HAVE_AS_NEEDED)
      target_link_libraries(torch_xpu PRIVATE
          "-Wl,--no-as-needed,\"$<TARGET_FILE:torch_cpu>\" -Wl,--as-needed")
    else()
      target_link_libraries(torch_xpu PRIVATE "$<TARGET_FILE:torch_cpu>")
    endif()
  endif()
endif()

# ---[ Metal(OSX) modification
if(APPLE AND USE_PYTORCH_METAL)
  if(NOT INTERN_BUILD_MOBILE)
    include(../cmake/Metal.cmake)
    # We need to link the system frameworks explicitly
    find_library(metal NAMES Metal)
    find_library(mps NAMES MetalPerformanceShaders)
    find_library(foundation NAMES Foundation)
    find_library(accelerate NAMES Accelerate)
    target_link_libraries(torch_cpu PUBLIC ${metal} ${mps} ${foundation} ${accelerate})
  endif()
endif()


target_link_libraries(torch_cpu PRIVATE flatbuffers)

# Note [Global dependencies]
# Some libraries (e.g. OpenMPI) like to dlopen plugins after they're initialized,
# and they assume that all of their symbols will be available in the global namespace.
# On the other hand we try to be good citizens and avoid polluting the symbol
# namespaces, so libtorch is loaded with all its dependencies in a local scope.
# That usually leads to missing symbol errors at run-time, so to avoid a situation like
# this we have to preload those libs in a global namespace.
if(BUILD_SHARED_LIBS)
  add_library(torch_global_deps SHARED ${TORCH_SRC_DIR}/csrc/empty.c)
  if(HAVE_SOVERSION)
    set_target_properties(torch_global_deps PROPERTIES
        VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
  endif()
  set_target_properties(torch_global_deps PROPERTIES LINKER_LANGUAGE C)
  if(USE_MPI)
    target_link_libraries(torch_global_deps MPI::MPI_CXX)
  endif()
  if(CAFFE2_USE_MKL)
    target_link_libraries(torch_global_deps caffe2::mkl)
  endif()
  # The CUDA libraries are linked here for a different reason: in some
  # cases we load these libraries with ctypes, and if they weren't opened
  # with RTLD_GLOBAL, we'll do the "normal" search process again (and
  # not find them, because they're usually in non-standard locations)
  if(USE_CUDA)
    target_link_libraries(torch_global_deps ${Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS})
    target_link_libraries(torch_global_deps torch::cudart)
    if(TARGET torch::nvtoolsext)
      target_link_libraries(torch_global_deps torch::nvtoolsext)
    endif()
  endif()
  install(TARGETS torch_global_deps DESTINATION "${TORCH_INSTALL_LIB_DIR}")
endif()

# ---[ Caffe2 HIP sources.
if(USE_ROCM)
  # Call again since Caffe2_HIP_INCLUDE is extended with ATen include dirs.
  # Get Compile Definitions from the directory (FindHIP.cmake bug)
  get_directory_property(MY_DEFINITIONS COMPILE_DEFINITIONS)
  if(MY_DEFINITIONS)
    foreach(_item ${MY_DEFINITIONS})
      list(APPEND HIP_CLANG_FLAGS "-D${_item}")
    endforeach()
  endif()

  # Call again since Caffe2_HIP_INCLUDE is extended with ATen include dirs.
  hip_include_directories(${Caffe2_HIP_INCLUDE})

  # Since PyTorch files contain HIP headers, these flags are required for the necessary definitions to be added.
  target_compile_options(torch_hip PUBLIC ${HIP_CXX_FLAGS})  # experiment

  target_link_libraries(torch_hip PUBLIC c10_hip)

  if(NOT INTERN_BUILD_MOBILE)
    # TODO: Cut this over to ATEN_HIP_FILES_GEN_LIB.  At the moment, we
    # only generate CUDA files
    # NB: This dependency must be PRIVATE, because we don't install
    # ATEN_CUDA_FILES_GEN_LIB (it's a synthetic target just to get the
    # correct dependency from generated files.)
    target_link_libraries(torch_hip PRIVATE ATEN_CUDA_FILES_GEN_LIB)
  endif()
  target_link_libraries(torch_hip PUBLIC torch_cpu_library ${Caffe2_PUBLIC_HIP_DEPENDENCY_LIBS})
  target_link_libraries(torch_hip PRIVATE ${Caffe2_HIP_DEPENDENCY_LIBS})

  # Since PyTorch files contain HIP headers, this is also needed to capture the includes.
  target_include_directories(torch_hip PRIVATE ${Caffe2_HIP_INCLUDE})
  target_include_directories(torch_hip INTERFACE $<INSTALL_INTERFACE:include>)
endif()

if(BUILD_STATIC_RUNTIME_BENCHMARK)
  add_subdirectory(${TORCH_ROOT}/benchmarks/static_runtime ${PROJECT_BINARY_DIR}/bin)
  add_executable(static_runtime_bench "${STATIC_RUNTIME_BENCHMARK_SRCS}")
  add_executable(static_runtime_test "${STATIC_RUNTIME_TEST_SRCS}")
  target_link_libraries(static_runtime_bench torch_library benchmark)
  target_link_libraries(static_runtime_test torch_library gtest_main)
endif()

if(BUILD_MOBILE_BENCHMARK)
  foreach(benchmark_src ${ATen_MOBILE_BENCHMARK_SRCS})
    get_filename_component(benchmark_name ${benchmark_src} NAME_WE)
    add_executable(${benchmark_name} "${benchmark_src}")
    target_link_libraries(${benchmark_name} torch_library benchmark)
    target_include_directories(${benchmark_name} PRIVATE $<INSTALL_INTERFACE:include>)
    target_include_directories(${benchmark_name} PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>)
    target_include_directories(${benchmark_name} PRIVATE ${ATen_CPU_INCLUDE})
    target_link_options(${benchmark_name} PRIVATE "LINKER:--allow-multiple-definition")
  endforeach()
endif()

if(BUILD_MOBILE_TEST)
  foreach(test_src ${ATen_MOBILE_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    target_link_libraries(${test_name} torch_library gtest_main)
    target_include_directories(${test_name} PRIVATE $<INSTALL_INTERFACE:include>)
    target_include_directories(${test_name} PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>)
    target_include_directories(${test_name} PRIVATE ${ATen_CPU_INCLUDE})
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
  endforeach()
endif()

# ---[ Test binaries.
if(BUILD_TEST)

  foreach(test_src ${ATen_VEC_TEST_SRCS})
    foreach(i RANGE ${NUM_CPU_CAPABILITY_NAMES})
        get_filename_component(test_name ${test_src} NAME_WE)
        list(GET CPU_CAPABILITY_NAMES ${i} CPU_CAPABILITY)
        list(GET CPU_CAPABILITY_FLAGS ${i} FLAGS)
        separate_arguments(FLAGS UNIX_COMMAND "${FLAGS}")
        # Build vec with minimal dependencies on all platforms but Windows
        if(NOT MSVC)
          add_executable(${test_name}_${CPU_CAPABILITY} "${test_src}" ../aten/src/ATen/native/quantized/AffineQuantizerBase.cpp)
          # TODO: Get rid of c10 dependency (which is only needed for the implementation of AT_ERROR)
          target_link_libraries(${test_name}_${CPU_CAPABILITY} c10 sleef gtest_main nlohmann)
          if(USE_FBGEMM)
            target_link_libraries(${test_name}_${CPU_CAPABILITY} fbgemm)
          endif()
          if(USE_ASAN)
            if(TARGET Sanitizer::address)
              target_link_libraries(${test_name}_${CPU_CAPABILITY} Sanitizer::address)
            endif()
            if(TARGET Sanitizer::undefined)
              target_link_libraries(${test_name}_${CPU_CAPABILITY} Sanitizer::undefined)
            endif()
          endif()
        else()
          add_executable(${test_name}_${CPU_CAPABILITY} "${test_src}")
          target_link_libraries(${test_name}_${CPU_CAPABILITY} torch_library gtest_main)
          if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
            target_link_libraries(${test_name}_${CPU_CAPABILITY} sleef)
          endif()
        endif()
        target_include_directories(${test_name}_${CPU_CAPABILITY} PRIVATE $<INSTALL_INTERFACE:include>)
        target_include_directories(${test_name}_${CPU_CAPABILITY} PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>)
        target_include_directories(${test_name}_${CPU_CAPABILITY} PRIVATE ${ATen_CPU_INCLUDE})
        target_compile_definitions(${test_name}_${CPU_CAPABILITY} PRIVATE CPU_CAPABILITY=${CPU_CAPABILITY}  CPU_CAPABILITY_${CPU_CAPABILITY})
        target_compile_options(${test_name}_${CPU_CAPABILITY} PRIVATE  ${FLAGS})
        if(NOT MSVC)
              target_compile_options(${test_name}_${CPU_CAPABILITY} PRIVATE -Wno-ignored-qualifiers)
        endif(NOT MSVC)
        add_test(NAME ${test_name}_${CPU_CAPABILITY} COMMAND $<TARGET_FILE:${test_name}_${CPU_CAPABILITY}>)
    endforeach()
  endforeach()

  foreach(test_src ${Caffe2_CPU_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    target_link_libraries(${test_name} torch_library gtest_main)
    if(NOT MSVC)
      target_link_libraries(${test_name} stdc++)
    endif()
    target_include_directories(${test_name} PRIVATE $<INSTALL_INTERFACE:include>)
    target_include_directories(${test_name} PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>)
    target_include_directories(${test_name} PRIVATE ${Caffe2_CPU_INCLUDE})
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    if(INSTALL_TEST)
        set_target_properties(${test_name} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
      install(TARGETS ${test_name} DESTINATION test)
      # Install PDB files for MSVC builds
      if(MSVC AND BUILD_SHARED_LIBS)
        install(FILES $<TARGET_PDB_FILE:${test_name}> DESTINATION test OPTIONAL)
      endif()
    endif()
  endforeach()

  if(USE_MPS)
    foreach(test_src ${Caffe2_MPS_TEST_SRCS})
      get_filename_component(test_name ${test_src} NAME_WE)
      add_executable(${test_name} "${test_src}")
      find_library(metal NAMES Metal)
      find_library(foundation NAMES Foundation)
      target_link_libraries(${test_name} torch_library gtest_main ${metal} ${foundation})
      target_include_directories(${test_name} PRIVATE $<INSTALL_INTERFACE:include>)
      target_include_directories(${test_name} PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>)
      target_include_directories(${test_name} PRIVATE ${Caffe2_CPU_INCLUDE})
      add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
      if(INSTALL_TEST)
        set_target_properties(${test_name} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
        install(TARGETS ${test_name} DESTINATION test)
        # Install PDB files for MSVC builds
        if(MSVC AND BUILD_SHARED_LIBS)
          install(FILES $<TARGET_PDB_FILE:${test_name}> DESTINATION test OPTIONAL)
        endif()
      endif()
    endforeach()
  endif()

  if(USE_CUDA)
    foreach(test_src ${Caffe2_GPU_TEST_SRCS})
      get_filename_component(test_name ${test_src} NAME_WE)
      add_executable(${test_name} "${test_src}")
      target_link_libraries(${test_name} torch_library gtest_main)
      if(USE_CUDNN AND ${test_name} MATCHES "cudnn")
        target_link_libraries(${test_name} torch::cudnn)
      endif()
      target_include_directories(${test_name} PRIVATE $<INSTALL_INTERFACE:include>)
      target_include_directories(${test_name} PRIVATE ${Caffe2_CPU_INCLUDE})
      add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
      if(INSTALL_TEST)
      set_target_properties(${test_name} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
        install(TARGETS ${test_name} DESTINATION test)
        # Install PDB files for MSVC builds
        if(MSVC AND BUILD_SHARED_LIBS)
          install(FILES $<TARGET_PDB_FILE:${test_name}> DESTINATION test OPTIONAL)
        endif()
      endif()
    endforeach()
    if(TARGET context_gpu_test)
      target_link_libraries(context_gpu_test caffe2::curand caffe2::cublas)
    endif()
  endif()

  if(USE_XPU)
    foreach(test_src ${Caffe2_XPU_TEST_SRCS})
      get_filename_component(test_name ${test_src} NAME_WE)
      add_executable(${test_name} "${test_src}")
      target_link_libraries(${test_name} torch_library gtest_main)
      target_include_directories(${test_name} PRIVATE $<INSTALL_INTERFACE:include>)
      target_include_directories(${test_name} PRIVATE ${Caffe2_CPU_INCLUDE})
      add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
      if(INSTALL_TEST)
        set_target_properties(${test_name} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
        install(TARGETS ${test_name} DESTINATION test)
      endif()
    endforeach()
  endif()

  if(USE_VULKAN)
    foreach(test_src ${Caffe2_VULKAN_TEST_SRCS})
      get_filename_component(test_name ${test_src} NAME_WE)
      add_executable(${test_name} "${test_src}")
      target_link_libraries(${test_name} torch_library gtest_main)
      target_include_directories(${test_name} PRIVATE $<INSTALL_INTERFACE:include>)
      target_include_directories(${test_name} PRIVATE ${Caffe2_CPU_INCLUDE})
      add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
      if(INSTALL_TEST)
        set_target_properties(${test_name} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
        install(TARGETS ${test_name} DESTINATION test)
        # Install PDB files for MSVC builds
        if(MSVC AND BUILD_SHARED_LIBS)
          install(FILES $<TARGET_PDB_FILE:${test_name}> DESTINATION test OPTIONAL)
        endif()
      endif()
    endforeach()
  endif()

  if(USE_ROCM)
    foreach(test_src ${Caffe2_HIP_TEST_SRCS})
      get_filename_component(test_name ${test_src} NAME_WE)
      add_executable(${test_name} "${test_src}")
      target_link_libraries(${test_name} torch_library gtest_main)
      target_include_directories(${test_name} PRIVATE $<INSTALL_INTERFACE:include>)
      target_include_directories(${test_name} PRIVATE ${Caffe2_CPU_INCLUDE} ${Caffe2_HIP_INCLUDE})
      target_compile_options(${test_name} PRIVATE ${HIP_CXX_FLAGS})
      add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
      if(INSTALL_TEST)
        set_target_properties(${test_name} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
        install(TARGETS ${test_name} DESTINATION test)
      endif()
    endforeach()
  endif()
endif()

if(MSVC)
  # This is used to enable the conforming lambda processor in MSVC
  # Which allows us to capture constexpr in lambdas
  # Note that this will be turned on by default for std=c++20 and above
  # This should be applied globally when https://github.com/pytorch/pytorch/issues/92600 is fixed
  foreach(tmp ${MEM_EFF_ATTENTION_CUDA_SOURCES})
    # MEM_EFF_ATTENTION_CUDA is populated in pytorch/aten/src/ATen/CMakeLists.txt
    # We iterate over these files, updating paths and adding the compile flag
    FILE(RELATIVE_PATH tmp_path "${PROJECT_SOURCE_DIR}" "${tmp}")
    SET(tmp_path "../${tmp_path}")
    set_source_files_properties(${tmp_path} PROPERTIES COMPILE_FLAGS "-Xcompiler /Zc:lambda")
  endforeach()
endif()
endif()
