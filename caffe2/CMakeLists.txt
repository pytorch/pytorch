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
#  TBB - using TBB for intra- and native thread pool for inter-op parallelism
if(INTERN_BUILD_MOBILE)
  set(ATEN_THREADING "NATIVE" CACHE STRING "ATen parallel backend")
else()
  if(USE_OPENMP)
    set(ATEN_THREADING "OMP" CACHE STRING "ATen parallel backend")
  elseif(USE_TBB)
    set(ATEN_THREADING "TBB" CACHE STRING "ATen parallel backend")
  else()
    set(ATEN_THREADING "NATIVE" CACHE STRING "ATen parallel backend")
  endif()
endif()

set(AT_PARALLEL_OPENMP 0)
set(AT_PARALLEL_NATIVE 0)
set(AT_PARALLEL_NATIVE_TBB 0)

message(STATUS "Using ATen parallel backend: ${ATEN_THREADING}")
if("${ATEN_THREADING}" STREQUAL "OMP")
  set(AT_PARALLEL_OPENMP 1)
elseif("${ATEN_THREADING}" STREQUAL "NATIVE")
  set(AT_PARALLEL_NATIVE 1)
elseif("${ATEN_THREADING}" STREQUAL "TBB")
  if(NOT USE_TBB)
    message(FATAL_ERROR "Using TBB backend but USE_TBB is off")
  endif()
  message(WARNING "ATEN TBB Threading is deprectated.")
  set(AT_PARALLEL_NATIVE_TBB 1)
else()
  message(FATAL_ERROR "Unknown ATen parallel backend: ${ATEN_THREADING}")
endif()

# ---[ Declare source file lists

# ---[ ATen build
if(INTERN_BUILD_ATEN_OPS)
  set(__caffe2_CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE})
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_subdirectory(../aten aten)
  set(CMAKE_POSITION_INDEPENDENT_CODE ${__caffe2_CMAKE_POSITION_INDEPENDENT_CODE})

  # Generate the headers wrapped by our operator
  file(GLOB_RECURSE torchgen_python "${PROJECT_SOURCE_DIR}/torchgen/*.py")
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/contrib/aten/aten_op.h
  COMMAND
  "${PYTHON_EXECUTABLE}" ${CMAKE_CURRENT_SOURCE_DIR}/contrib/aten/gen_op.py
    --aten_root=${CMAKE_CURRENT_SOURCE_DIR}/../aten
    --template_dir=${CMAKE_CURRENT_SOURCE_DIR}/contrib/aten
    --yaml_dir=${CMAKE_CURRENT_BINARY_DIR}/../aten/src/ATen
    --install_dir=${CMAKE_CURRENT_BINARY_DIR}/contrib/aten
  DEPENDS
  ${torchgen_python}
  ${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml
  ${CMAKE_CURRENT_SOURCE_DIR}/contrib/aten/gen_op.py
  ${CMAKE_CURRENT_SOURCE_DIR}/contrib/aten/aten_op_template.h)

  add_custom_target(__aten_op_header_gen
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/contrib/aten/aten_op.h)
  add_library(aten_op_header_gen INTERFACE)
  add_dependencies(aten_op_header_gen __aten_op_header_gen)

  # Add source, includes, and libs to lists
  list(APPEND Caffe2_CPU_SRCS ${ATen_CPU_SRCS})
  list(APPEND Caffe2_GPU_SRCS ${ATen_CUDA_CPP_SRCS})
  list(APPEND Caffe2_GPU_SRCS_W_SORT_BY_KEY ${ATen_CUDA_SRCS_W_SORT_BY_KEY})
  list(APPEND Caffe2_GPU_CU_SRCS ${ATen_CUDA_CU_SRCS})
  list(APPEND Caffe2_GPU_CU_SRCS_W_SORT_BY_KEY ${ATen_CUDA_CU_SRCS_W_SORT_BY_KEY})
  list(APPEND Caffe2_HIP_SRCS ${ATen_HIP_SRCS})
  list(APPEND Caffe2_MPS_SRCS ${ATen_MPS_SRCS})
  list(APPEND Caffe2_HIP_SRCS ${ATen_HIP_SRCS_W_SORT_BY_KEY})
  list(APPEND Caffe2_CPU_TEST_SRCS ${ATen_CPU_TEST_SRCS})
  list(APPEND Caffe2_MPS_TEST_SRCS ${ATen_MPS_TEST_SRCS})
  list(APPEND Caffe2_GPU_TEST_SRCS ${ATen_CUDA_TEST_SRCS})
  list(APPEND Caffe2_HIP_TEST_SRCS ${ATen_HIP_TEST_SRCS})
  list(APPEND Caffe2_CPU_TEST_SRCS ${ATen_CORE_TEST_SRCS})
  list(APPEND Caffe2_VULKAN_TEST_SRCS ${ATen_VULKAN_TEST_SRCS})
  list(APPEND Caffe2_CPU_INCLUDE ${ATen_CPU_INCLUDE})
  list(APPEND Caffe2_GPU_INCLUDE ${ATen_CUDA_INCLUDE})
  list(APPEND Caffe2_HIP_INCLUDE ${ATen_HIP_INCLUDE})
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
if(BUILD_CAFFE2 OR (NOT USE_FBGEMM))
  add_subdirectory(perfkernels)
endif()

# Skip modules that are not used by libtorch mobile yet.
if(BUILD_CAFFE2 AND NOT INTERN_BUILD_MOBILE)
  add_subdirectory(contrib)
  add_subdirectory(predictor)
  add_subdirectory(predictor/emulator)
  add_subdirectory(core/nomnigraph)
  if(USE_NVRTC)
    add_subdirectory(cuda_rtc)
  endif()
  add_subdirectory(db)
  add_subdirectory(distributed)
  # add_subdirectory(experiments) # note, we may remove this folder at some point
  add_subdirectory(ideep)
  add_subdirectory(image)
  add_subdirectory(video)
  add_subdirectory(mobile)
  add_subdirectory(mpi)
  add_subdirectory(observers)
  add_subdirectory(onnx)
  if(BUILD_CAFFE2_OPS)
    add_subdirectory(operators)
    add_subdirectory(operators/rnn)
    if(USE_FBGEMM)
      add_subdirectory(quantization/server)
    endif()
    if(USE_QNNPACK)
      add_subdirectory(operators/quantized)
    endif()
  endif()
  add_subdirectory(opt)
  add_subdirectory(proto)
  add_subdirectory(python)
  add_subdirectory(queue)
  add_subdirectory(sgd)
  add_subdirectory(share)
  # add_subdirectory(test) # todo: use caffe2_gtest_main instead of gtest_main because we will need to call GlobalInit
  add_subdirectory(transforms)
endif()
if(NOT BUILD_CAFFE2 AND NOT INTERN_BUILD_MOBILE)
  add_subdirectory(proto)
endif()

# Advanced: if we have allow list specified, we will do intersections for all
# main lib srcs.
if(CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_CPU_SRCS CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_GPU_SRCS CAFFE2_ALLOWLISTED_FILES)
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

  message(STATUS "ATen Vulkan test sources: ")
  foreach(tmp ${ATen_VULKAN_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

endif()

if(NOT INTERN_BUILD_MOBILE)
  # ---[ List of libraries to link with
  add_library(caffe2_protos STATIC $<TARGET_OBJECTS:Caffe2_PROTO>)
  add_dependencies(caffe2_protos Caffe2_PROTO)
  # If we are going to link protobuf locally inside caffe2 libraries, what we will do is
  # to create a helper static library that always contains libprotobuf source files, and
  # link the caffe2 related dependent libraries to it.
  target_include_directories(caffe2_protos INTERFACE $<INSTALL_INTERFACE:include>)
  # Reason for this public dependency is as follows:
  # (1) Strictly speaking, we should not expose any Protobuf related functions. We should
  #     only use function interfaces wrapped with our own public API, and link protobuf
  #     locally.
  # (2) However, currently across the Caffe2 codebase, we have extensive use of protobuf
  #     functionalities. For example, not only libcaffe2.so uses it, but also other
  #     binaries such as python extensions etc. As a result, we will have to have a
  #     transitive dependency to libprotobuf.
  #
  # Good thing is that, if we specify CAFFE2_LINK_LOCAL_PROTOBUF, then we do not need to
  # separately deploy protobuf binaries - libcaffe2.so will contain all functionalities
  # one needs. One can verify this via ldd.
  #
  # TODO item in the future includes:
  # (1) Enable using lite protobuf
  # (2) Properly define public API that do not directly depend on protobuf itself.
  # (3) Expose the libprotobuf.a file for dependent libraries to link to.
  #
  # What it means for users/developers?
  # (1) Users: nothing affecting the users, other than the fact that CAFFE2_LINK_LOCAL_PROTOBUF
  #     avoids the need to deploy protobuf.
  # (2) Developers: if one simply uses core caffe2 functionality without using protobuf,
  #     nothing changes. If one has a dependent library that uses protobuf, then one needs to
  #     have the right protobuf version as well as linking to libprotobuf.a.
  target_link_libraries(caffe2_protos PUBLIC protobuf::libprotobuf)
  if(NOT BUILD_SHARED_LIBS)
    install(TARGETS caffe2_protos ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  endif()
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
  )

set(GENERATED_TESTING_PYTHON
  "${TORCH_SRC_DIR}/testing/_internal/generated/annotated_fn_args.py"
  )

set(TORCH_GENERATED_CODE
  ${GENERATED_CXX_TORCH}
  ${GENERATED_H_TORCH}
  ${GENERATED_CXX_PYTHON}
  ${GENERATED_H_PYTHON}
  ${GENERATED_TESTING_PYTHON}
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
  "${PYTHON_EXECUTABLE}" tools/setup_helpers/generate_code.py
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
  # Force -Werror on several files
  set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/mkldnn/Pooling.cpp PROPERTIES COMPILE_FLAGS "-Werror")
endif()
# Disable certain warnings for GCC-9.X
if(CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0.0))
  # See https://github.com/pytorch/pytorch/issues/38856
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/tensorexpr/llvm_jit.cpp PROPERTIES COMPILE_FLAGS "-Wno-redundant-move -Wno-noexcept-type")
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/tensorexpr/llvm_codegen.cpp PROPERTIES COMPILE_FLAGS "-Wno-init-list-lifetime")
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

  # Disable legacy import of building without Caffe2 support
  if(BUILD_CAFFE2)
    list(APPEND TORCH_SRCS
      ${TORCH_SRC_DIR}/csrc/jit/serialization/import_legacy.cpp
    )
  else()
    set_source_files_properties(
      ${TORCH_SRC_DIR}/csrc/jit/serialization/import.cpp
      PROPERTIES COMPILE_FLAGS "-DC10_DISABLE_LEGACY_IMPORT"
    )
  endif()
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

  target_link_libraries(caffe2_nvrtc ${CUDA_NVRTC} ${CUDA_CUDA_LIB} ${CUDA_NVRTC_LIB} ${DELAY_LOAD_FLAGS})
  target_include_directories(caffe2_nvrtc PRIVATE ${CUDA_INCLUDE_DIRS})
  install(TARGETS caffe2_nvrtc DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  if(USE_NCCL)
    list(APPEND Caffe2_GPU_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/nccl.cpp)
  endif()
  if(USE_DISTRIBUTED)
    append_filelist("libtorch_cuda_distributed_base_sources" Caffe2_GPU_SRCS)
    if(NOT WIN32)
      append_filelist("libtorch_cuda_distributed_extra_sources" Caffe2_GPU_SRCS)
    endif()
  endif()
  set_source_files_properties(
    ${TORCH_ROOT}/aten/src/ATen/cuda/detail/LazyNVRTC.cpp
    PROPERTIES COMPILE_DEFINITIONS "NVRTC_SHORTHASH=${CUDA_NVRTC_SHORTHASH}"
  )
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/passes/frozen_conv_add_relu_fusion.cpp PROPERTIES COMPILE_FLAGS "-DUSE_CUDA=1")
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/interface.cpp PROPERTIES COMPILE_FLAGS "-DUSE_CUDA=1")
endif()

if(USE_FLASH_ATTENTION AND NOT MSVC)
  # Cutlass contains a sign-compare violation in its codebase to be fixed by https://github.com/NVIDIA/cutlass/pull/869
  set_source_files_properties(
    "${PROJECT_SOURCE_DIR}/aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu"
    "${PROJECT_SOURCE_DIR}/aten/src/ATen/native/nested/cuda/NestedTensorTransformerFunctions.cu"
    "${PROJECT_SOURCE_DIR}/aten/src/ATen/native/transformers/cuda/flash_attn/fmha_bwd_hdim128.cu"
    PROPERTIES COMPILE_FLAGS "-Wno-sign-compare")
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
  target_link_libraries(caffe2_nvrtc ${PYTORCH_HIP_HCC_LIBRARIES} ${ROCM_HIPRTC_LIB})
  target_compile_definitions(caffe2_nvrtc PRIVATE USE_ROCM __HIP_PLATFORM_HCC__)
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
  )
endif()

list(APPEND Caffe2_CPU_SRCS ${TORCH_SRCS})

if(USE_MPS)
  list(APPEND Caffe2_CPU_SRCS ${Caffe2_MPS_SRCS})
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

add_library(torch_cpu ${Caffe2_CPU_SRCS})
if(HAVE_SOVERSION)
  set_target_properties(torch_cpu PROPERTIES
      VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
endif()
torch_compile_options(torch_cpu)  # see cmake/public/utils.cmake
if(BUILD_CAFFE2 AND NOT MSVC)
  # Caffe2 has too many signed-unsigned violation, but the framework is dead
  # So no point in fixing those
  target_compile_options(torch_cpu PRIVATE "-Wno-sign-compare")
endif()

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  target_compile_options_if_supported(torch_cpu "-Wmissing-prototypes")
  target_compile_options_if_supported(torch_cpu "-Werror=missing-prototypes")
  set(ignored_files "${CMAKE_CURRENT_SOURCE_DIR}/utils/proto_wrap.cc")
  foreach(generated_file IN LISTS GENERATED_CXX_TORCH ignored_files ATen_MPS_SRCS)
    set_source_files_properties(${generated_file} PROPERTIES COMPILE_OPTIONS "-Wno-missing-prototypes;-Wno-error=missing-prototypes")
  endforeach()
  foreach(source_file IN LISTS ATen_CPU_SRCS)
      string(FIND "${source_file}" "metal" res)
      if(res GREATER -1)
          set_source_files_properties(${source_file} PROPERTIES COMPILE_OPTIONS "-Wno-missing-prototypes;-Wno-error=missing-prototypes")
      endif()
  endforeach()
  foreach(source_file IN LISTS Caffe2_CPU_SRCS)
    string(FIND "${source_file}" "${CMAKE_BINARY_DIR}" res)
    if(res GREATER -1)
      set_source_files_properties(${source_file} PROPERTIES COMPILE_OPTIONS "-Wno-missing-prototypes;-Wno-error=missing-prototypes")
    endif()
  endforeach()
endif()

set_property(SOURCE ${ATen_CORE_SRCS} APPEND
    PROPERTY COMPILE_DEFINITIONS "TORCH_ASSERT_ONLY_METHOD_OPERATORS")

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
  hip_add_library(torch_hip ${Caffe2_HIP_SRCS})
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
endif()

if(NOT MSVC AND USE_XNNPACK)
  TARGET_LINK_LIBRARIES(torch_cpu PRIVATE fxdiv)
endif()

# ==========================================================
# formerly-libtorch flags
# ==========================================================

if(NOT INTERN_BUILD_MOBILE)
  # Forces caffe2.pb.h to be generated before its dependents are compiled.
  # Adding the generated header file to the ${TORCH_SRCS} list is not sufficient
  # to establish the dependency, since the generation procedure is declared in a different CMake file.
  # See https://samthursfield.wordpress.com/2015/11/21/cmake-dependencies-between-targets-and-files-and-custom-commands/#custom-commands-in-different-directories
  add_dependencies(torch_cpu Caffe2_PROTO)
endif()

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
        "${PYTHON_EXECUTABLE}"
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
        "${PYTHON_EXECUTABLE}"
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

if(USE_TBB)
  list(APPEND ATen_CPU_INCLUDE ${TBB_INCLUDE_DIR})
  target_link_libraries(torch_cpu PUBLIC TBB::tbb)
endif()

if(BUILD_CAFFE2 AND BUILD_CAFFE2_OPS AND USE_FBGEMM)
  # FIXME: quantization/server/conv_dnnlowp_op.cc depends on fbgemm/src/RefImplementations.h
  target_include_directories(torch_cpu PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../third_party)
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
      ${TORCH_ROOT}/test/cpp/aot_inductor
      ${CMAKE_BINARY_DIR}/test_aot_inductor)
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
    __HIP_PLATFORM_HCC__
    )
  # NB: Massive hack.  torch/csrc/jit/codegen/fuser/codegen.cpp includes
  # torch/csrc/jit/codegen/fuser/cuda/resource_strings.h which changes the
  # strings depending on if you're __HIP_PLATFORM_HCC__ or not.
  # But that file is in torch_cpu!  So, against all odds, this macro
  # has to be set on torch_cpu too.  I also added it to torch for
  # better luck
  target_compile_definitions(torch_cpu PRIVATE
    USE_ROCM
    __HIP_PLATFORM_HCC__
    )
  target_compile_definitions(torch PRIVATE
    USE_ROCM
    __HIP_PLATFORM_HCC__
    )
  target_include_directories(torch_hip PRIVATE
    /opt/rocm/include
    /opt/rocm/hcc/include
    /opt/rocm/rocblas/include
    /opt/rocm/hipsparse/include
    )
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
      if(USE_NCCL_WITH_UCC)
        target_compile_definitions(torch_cuda PUBLIC USE_NCCL_WITH_UCC)
      endif()
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
  caffe2_interface_library(caffe2_protos caffe2_protos_whole)
  target_link_libraries(torch_cpu PRIVATE caffe2_protos_whole)
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
target_include_directories(torch_cpu INTERFACE $<INSTALL_INTERFACE:include>)
target_include_directories(torch_cpu PRIVATE ${Caffe2_CPU_INCLUDE})
target_include_directories(torch_cpu SYSTEM PRIVATE "${Caffe2_DEPENDENCY_INCLUDE}")

target_compile_definitions(torch_cpu PRIVATE CAFFE2_BUILD_MAIN_LIB)
if(USE_CUDA)
  target_compile_definitions(torch_cuda PRIVATE TORCH_CUDA_BUILD_MAIN_LIB)
elseif(USE_ROCM)
  target_compile_definitions(torch_hip PRIVATE TORCH_HIP_BUILD_MAIN_LIB)
endif()

if(USE_EXPERIMENTAL_CUDNN_V8_API)
  if(USE_CUDA)
    target_compile_definitions(torch_cuda PRIVATE "-DUSE_EXPERIMENTAL_CUDNN_V8_API")
  endif()
endif()

set(EXPERIMENTAL_SINGLE_THREAD_POOL "0" CACHE STRING
  "Experimental option to use a single thread pool for inter- and intra-op parallelism")
if("${EXPERIMENTAL_SINGLE_THREAD_POOL}")
  target_compile_definitions(torch_cpu PUBLIC "-DAT_EXPERIMENTAL_SINGLE_THREAD_POOL=1")
endif()

if(MSVC AND NOT BUILD_SHARED_LIBS)
  # Note [Supporting both static and dynamic libraries on Windows]
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # A Windows library may be distributed as either a static or dynamic
  # library.  The chosen distribution mechanism affects how you setup
  # the headers for the library: if you statically link a function,
  # all you need is an ordinary signature:
  #
  #     void f();
  #
  # But if you *dynamically* link it, then you must provide a __declspec
  # specifying that it should be imported from a DLL:
  #
  #     __declspec(dllimport) void f();
  #
  # Mixing the two situations will not work: if you specify dllimport
  # while statically linking, the linker will complain it cannot find
  # the __imp_f symbol (which serve as the DLL entrypoint); if you
  # fail to specify dllimport for a symbol that's coming from a DLL,
  # the linker will complain that it can't find f.  Joy!
  #
  # Most places on the Internet, you will find people have written
  # their headers under the assumption that the application will
  # only ever be dynamically linked, as they define a macro which
  # tags a function as __declspec(dllexport) if you are actually
  # building the library, and __declspec(dllimport) otherwise.  But
  # if you want these headers to also work if you are linking against
  # a static library, you need a way to avoid adding these __declspec's
  # at all.  And that "mechanism" needs to apply to any downstream
  # libraries/executables which are going to link against your library.
  #
  #   As an aside, why do we need to support both modes?
  #   For historical reasons, PyTorch ATen on Windows is built dynamically,
  #   while Caffe2 on Windows is built statically (mostly because if
  #   we build it dynamically, we are over the DLL exported symbol limit--and
  #   that is because Caffe2 hasn't comprehensively annotated all symbols
  #   which cross the DLL boundary with CAFFE_API).  So any code
  #   which is used by both PyTorch and Caffe2 needs to support both
  #   modes of linking.
  #
  # So, you have a macro (call it AT_CORE_STATIC_WINDOWS) which you need to have
  # set for any downstream library/executable that transitively includes your
  # headers.  How are you going to do this?  You have two options:
  #
  #   1. Write out a config.h header which stores whether or not
  #      you are linking statically or dynamically.
  #
  #   2. Force all of users to set the the macro themselves.  If they
  #      use cmake, you can set -DAT_CORE_STATIC_WINDOWS=1 as a PUBLIC
  #      compile option, in which case cmake will automatically
  #      add the macro for you.
  #
  # Which one is better? Well, it depends: they trade off implementor
  # ease versus user ease: (1) is more work for the library author
  # but the user doesn't have to worry about it; (2) requires the user
  # to set the macro themselves... but only if they don't use cmake.
  #
  # So, which is appropriate in our situation?  In my mind, here is
  # the distinguishing factor: it is more common to distribute
  # DLLs, since they don't require you to line up the CRT version
  # (/MD, /MDd, /MT, /MTd) and MSVC version at the use site.  So,
  # if a user is already in the business of static linkage, they're
  # already in "expert user" realm.  So, I've decided that at this
  # point in time, the simplicity of implementation of (2) wins out.
  #
  # NB: This must be target_compile_definitions, not target_compile_options,
  # as the latter is not respected by nvcc
  target_compile_definitions(torch_cpu PUBLIC "AT_CORE_STATIC_WINDOWS=1")
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
endif()

caffe2_interface_library(torch torch_library)

install(TARGETS torch_cpu torch_cpu_library EXPORT Caffe2Targets DESTINATION "${TORCH_INSTALL_LIB_DIR}")

if(USE_CUDA)
  install(TARGETS torch_cuda torch_cuda_library EXPORT Caffe2Targets DESTINATION "${TORCH_INSTALL_LIB_DIR}")
elseif(USE_ROCM)
  install(TARGETS torch_hip torch_hip_library EXPORT Caffe2Targets DESTINATION "${TORCH_INSTALL_LIB_DIR}")
endif()
install(TARGETS torch torch_library EXPORT Caffe2Targets DESTINATION "${TORCH_INSTALL_LIB_DIR}")

target_link_libraries(torch PUBLIC torch_cpu_library)

if(USE_CUDA)
  target_link_libraries(torch PUBLIC torch_cuda_library)
elseif(USE_ROCM)
  target_link_libraries(torch PUBLIC torch_hip_library)
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
  if(USE_KINETO AND NOT MSVC)
    target_link_libraries(torch_cpu PRIVATE torch::cudart)
  endif()
  target_link_libraries(torch_cuda INTERFACE torch::cudart)
  target_link_libraries(torch_cuda PUBLIC c10_cuda torch::nvtoolsext)

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
      target_link_libraries(torch_global_deps ${MPI_CXX_LIBRARIES})
  endif()
  target_link_libraries(torch_global_deps ${MKL_LIBRARIES})
  # The CUDA libraries are linked here for a different reason: in some
  # cases we load these libraries with ctypes, and if they weren't opened
  # with RTLD_GLOBAL, we'll do the "normal" search process again (and
  # not find them, because they're usually in non-standard locations)
  if(USE_CUDA)
    target_link_libraries(torch_global_deps ${Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS})
    target_link_libraries(torch_global_deps torch::cudart torch::nvtoolsext)
  endif()
  if(USE_TBB)
    target_link_libraries(torch_global_deps TBB::tbb)
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

if(BUILD_TENSOREXPR_BENCHMARK)
  add_subdirectory(${TORCH_ROOT}/benchmarks/cpp/tensorexpr ${CMAKE_BINARY_DIR}/tensorexpr_bench)
endif()

if(BUILD_CPP_BENCHMARKS)
  add_subdirectory(${TORCH_ROOT}/benchmarks/cpp ${PROJECT_BINARY_DIR}/bin)
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
          target_link_libraries(${test_name}_${CPU_CAPABILITY} c10 sleef gtest_main)
          if(USE_FBGEMM)
            target_link_libraries(${test_name}_${CPU_CAPABILITY} fbgemm)
          endif()
        else()
          add_executable(${test_name}_${CPU_CAPABILITY} "${test_src}")
          target_link_libraries(${test_name}_${CPU_CAPABILITY} torch_library gtest_main)
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
    target_include_directories(${test_name} PRIVATE $<INSTALL_INTERFACE:include>)
    target_include_directories(${test_name} PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>)
    target_include_directories(${test_name} PRIVATE ${Caffe2_CPU_INCLUDE})
    if(NOT MSVC)
      target_compile_options(${test_name} PRIVATE -Wno-unused-variable)
    endif()
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    if(INSTALL_TEST)
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
      target_include_directories(${test_name} PRIVATE $<INSTALL_INTERFACE:include>)
      target_include_directories(${test_name} PRIVATE ${Caffe2_CPU_INCLUDE})
      add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
      if(INSTALL_TEST)
        install(TARGETS ${test_name} DESTINATION test)
        # Install PDB files for MSVC builds
        if(MSVC AND BUILD_SHARED_LIBS)
          install(FILES $<TARGET_PDB_FILE:${test_name}> DESTINATION test OPTIONAL)
        endif()
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
        install(TARGETS ${test_name} DESTINATION test)
      endif()
    endforeach()
  endif()

  # For special tests that explicitly uses dependencies, we add them here
  if(BUILD_CAFFE2 AND USE_MPI)
    target_link_libraries(mpi_test ${MPI_CXX_LIBRARIES})
    if(USE_CUDA)
      target_link_libraries(mpi_gpu_test ${MPI_CXX_LIBRARIES})
    endif()
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

# Note: we only install the caffe2 python files if BUILD_CAFFE2_OPS is ON
# This is because the build rules here written in such a way that they always
# appear to need to be re-run generating >600 pieces of work during the pytorch
# rebuild step. The long-term fix should be to clean up these rules so they
# only rerun when needed.

if(BUILD_PYTHON)
  # Python site-packages
  # Get canonical directory for python site packages (relative to install
  # location).  It varies from system to system.
  # We should pin the path separator to the forward slash on Windows.
  # More details can be seen at
  # https://github.com/pytorch/pytorch/tree/main/tools/build_pytorch_libs.bat#note-backslash-munging-on-windows
  pycmd(PYTHON_SITE_PACKAGES "
      import os
      import sysconfig
      relative_site_packages = sysconfig.get_path('purelib').replace(sysconfig.get_path('data'), '').lstrip(os.path.sep)
      print(relative_site_packages)
  ")
  file(TO_CMAKE_PATH ${PYTHON_SITE_PACKAGES} PYTHON_SITE_PACKAGES)
  set(PYTHON_SITE_PACKAGES ${PYTHON_SITE_PACKAGES} PARENT_SCOPE) # for Summary
  # ---[ Options.
  set(PYTHON_LIB_REL_PATH "${PYTHON_SITE_PACKAGES}" CACHE STRING "Python installation path (relative to CMake installation prefix)")
  message(STATUS "Using ${PYTHON_LIB_REL_PATH} as python relative installation path")
  # Python extension suffix
  # Try to get from python through sysconfig.get_env_var('EXT_SUFFIX') first,
  # fallback to ".pyd" if windows and ".so" for all others.
  pycmd(PY_EXT_SUFFIX "
      def get_ext_suffix():
          import sys
          import sysconfig
          return sysconfig.get_config_var('EXT_SUFFIX')

      suffix = get_ext_suffix()
      if suffix is not None:
          print(suffix)
      else:
          print()
  ")
  if("${PY_EXT_SUFFIX}" STREQUAL "")
    if(MSVC)
      set(PY_EXT_SUFFIX ".pyd")
    else()
      set(PY_EXT_SUFFIX ".so")
    endif()
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947 in EmbeddingBag.cpp
    set_source_files_properties(../aten/src/ATen/native/EmbeddingBag.cpp PROPERTIES COMPILE_FLAGS -Wno-attributes)
    set_source_files_properties(${TORCH_SRC_DIR}/../caffe2/operators/box_with_nms_limit_op.cc PROPERTIES COMPILE_FLAGS -Wno-attributes)
  endif()
  # Allow different install locations for libcaffe2
  # For setuptools installs (that all build Python), install libcaffe2 into
  # site-packages, alongside the torch libraries. The pybind11 library needs
  # an rpath to the torch library folder
  # For cmake installs, including c++ only installs, install libcaffe2 into
  # CMAKE_INSTALL_PREFIX/lib . The pybind11 library can have a hardcoded
  # rpath
  set(caffe2_pybind11_rpath "${_rpath_portable_origin}")
  if(${BUILDING_WITH_TORCH_LIBS})
    # site-packages/caffe2/python/caffe2_pybind11_state
    # site-packages/torch/lib
    set(caffe2_pybind11_rpath "${_rpath_portable_origin}/../../torch/lib")
  endif(${BUILDING_WITH_TORCH_LIBS})

  # Must also include `CMAKE_SHARED_LINKER_FLAGS` in linker flags for
  # `caffe2_pybind11_state_*` targets because paths to required libraries may
  # need to be found there (e.g., specifying path to `libiomp5` with `LDFLAGS`).
  set(_caffe2_pybind11_state_linker_flags "${CMAKE_SHARED_LINKER_FLAGS}")
  if(APPLE)
    set(_caffe2_pybind11_state_linker_flags "${_caffe2_pybind11_state_linker_flags} -undefined dynamic_lookup")
  endif()

  # ---[ Python.
  if(BUILD_CAFFE2)
  add_library(caffe2_pybind11_state MODULE ${Caffe2_CPU_PYTHON_SRCS})
  target_compile_definitions(torch PRIVATE BUILD_CAFFE2)
  target_compile_definitions(torch_python PRIVATE BUILD_CAFFE2)
  if(USE_NUMPY)
    target_compile_options(caffe2_pybind11_state PRIVATE "-DUSE_NUMPY")
    target_link_libraries(caffe2_pybind11_state  PRIVATE numpy::numpy)
  endif()
  if(NOT MSVC)
    set_target_properties(caffe2_pybind11_state PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  endif()
  set_target_properties(caffe2_pybind11_state PROPERTIES PREFIX "" DEBUG_POSTFIX "")
  set_target_properties(caffe2_pybind11_state PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
  set_target_properties(caffe2_pybind11_state PROPERTIES LINK_FLAGS "${_caffe2_pybind11_state_linker_flags}")
  target_include_directories(caffe2_pybind11_state PRIVATE $<INSTALL_INTERFACE:include>)
  target_include_directories(caffe2_pybind11_state PRIVATE ${Caffe2_CPU_INCLUDE})

  target_link_libraries(caffe2_pybind11_state PRIVATE
      torch_library python::python pybind::pybind11)
  if(USE_MKLDNN)
      target_link_libraries(caffe2_pybind11_state PRIVATE caffe2::mkldnn)
  endif()
  if(WIN32)
    target_link_libraries(caffe2_pybind11_state PRIVATE onnx_proto)
  endif(WIN32)

  # Install caffe2_pybind11_state(_gpu|hip) in site-packages/caffe2/python,
  # so it needs an rpath to find libcaffe2
  set_target_properties(
      caffe2_pybind11_state PROPERTIES LIBRARY_OUTPUT_DIRECTORY
      ${CMAKE_BINARY_DIR}/caffe2/python)
  install(TARGETS caffe2_pybind11_state DESTINATION "${PYTHON_LIB_REL_PATH}/caffe2/python")
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:caffe2_pybind11_state> DESTINATION "${PYTHON_LIB_REL_PATH}/caffe2/python" OPTIONAL)
  endif()
  set_target_properties(caffe2_pybind11_state PROPERTIES INSTALL_RPATH "${caffe2_pybind11_rpath}")

  if(USE_CUDA)
    add_library(caffe2_pybind11_state_gpu MODULE ${Caffe2_GPU_PYTHON_SRCS})
    if(USE_NUMPY)
      target_compile_options(caffe2_pybind11_state_gpu PRIVATE "-DUSE_NUMPY")
      target_link_libraries(caffe2_pybind11_state_gpu PRIVATE numpy::numpy)
    endif()
    if(NOT MSVC)
      set_target_properties(caffe2_pybind11_state_gpu PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
    endif()
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES PREFIX "" DEBUG_POSTFIX "")
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES LINK_FLAGS "${_caffe2_pybind11_state_linker_flags}")
    target_include_directories(caffe2_pybind11_state_gpu PRIVATE $<INSTALL_INTERFACE:include>)
    target_include_directories(caffe2_pybind11_state_gpu PRIVATE ${Caffe2_CPU_INCLUDE})
    target_link_libraries(caffe2_pybind11_state_gpu PRIVATE
        torch_library python::python pybind::pybind11)
    if(WIN32)
      target_link_libraries(caffe2_pybind11_state_gpu PRIVATE onnx_proto)
    endif(WIN32)

    # Install with same rpath as non-gpu caffe2_pybind11_state
    set_target_properties(
        caffe2_pybind11_state_gpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/caffe2/python)
    install(TARGETS caffe2_pybind11_state_gpu DESTINATION "${PYTHON_LIB_REL_PATH}/caffe2/python")
    if(MSVC AND BUILD_SHARED_LIBS)
      install(FILES $<TARGET_PDB_FILE:caffe2_pybind11_state_gpu> DESTINATION "${PYTHON_LIB_REL_PATH}/caffe2/python" OPTIONAL)
    endif()
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES INSTALL_RPATH "${caffe2_pybind11_rpath}")
  endif()

  if(USE_ROCM)
    add_library(caffe2_pybind11_state_hip MODULE ${Caffe2_HIP_PYTHON_SRCS})
    if(USE_NUMPY)
      target_compile_options(caffe2_pybind11_state_hip PRIVATE "-DUSE_NUMPY")
      target_link_libraries(caffe2_pybind11_state_hip PRIVATE numpy::numpy)
    endif()
    if(NOT MSVC)
      target_compile_options(caffe2_pybind11_state_hip PRIVATE ${HIP_CXX_FLAGS} -fvisibility=hidden)
    endif()
    set_target_properties(caffe2_pybind11_state_hip PROPERTIES PREFIX "")
    set_target_properties(caffe2_pybind11_state_hip PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
    set_target_properties(caffe2_pybind11_state_hip PROPERTIES LINK_FLAGS "${_caffe2_pybind11_state_linker_flags}")
    target_include_directories(caffe2_pybind11_state_hip PRIVATE $<INSTALL_INTERFACE:include>)
    target_include_directories(caffe2_pybind11_state_hip PRIVATE ${Caffe2_CPU_INCLUDE} ${Caffe2_HIP_INCLUDE})
    target_link_libraries(caffe2_pybind11_state_hip PRIVATE
        torch_library python::python pybind::pybind11)

    # Install with same rpath as non-hip caffe2_pybind11_state
    set_target_properties(
        caffe2_pybind11_state_hip PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/caffe2/python)
    install(TARGETS caffe2_pybind11_state_hip DESTINATION "${PYTHON_LIB_REL_PATH}/caffe2/python")
    set_target_properties(caffe2_pybind11_state_hip PROPERTIES INSTALL_RPATH "${caffe2_pybind11_rpath}")
  endif()

  if(MSVC AND CMAKE_GENERATOR MATCHES "Visual Studio")
    # If we are building under windows, we will copy the file from
    # build/caffe2/python/{Debug,Release}/caffe2_pybind11_state.pyd
    # to its parent folder so that we can do in-build execution.
    add_custom_target(windows_python_copy_lib ALL)
    add_dependencies(windows_python_copy_lib caffe2_pybind11_state)
    add_custom_command(
        TARGET windows_python_copy_lib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:caffe2_pybind11_state>
        ${CMAKE_BINARY_DIR}/caffe2/python)
    if(USE_CUDA)
      add_dependencies(windows_python_copy_lib caffe2_pybind11_state_gpu)
      add_custom_command(
          TARGET windows_python_copy_lib POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:caffe2_pybind11_state_gpu>
          ${CMAKE_BINARY_DIR}/caffe2/python)
    endif()
    if(USE_ROCM)
      add_dependencies(windows_python_copy_lib caffe2_pybind11_state_hip)
      add_custom_command(
          TARGET windows_python_copy_lib POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:caffe2_pybind11_state_hip>
          ${CMAKE_BINARY_DIR}/caffe2/python)
    endif()
  endif()

  # Finally, Copy all python files to build directory
  # Create a custom target that copies all python files.
  file(GLOB_RECURSE PYTHON_SRCS RELATIVE ${PROJECT_SOURCE_DIR}
       "${PROJECT_SOURCE_DIR}/caffe2/*.py")
  endif()

  # generated pb files are copied from build/caffe2 to caffe2
  # if we copied them back to build this would create a build cycle
  # consider removing the need for globs
  filter_list_exclude(PYTHON_SRCS PYTHON_SRCS "proto/.*_pb")

  set(build_files)
  foreach(python_src ${PYTHON_SRCS})
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/${python_src}
                       DEPENDS ${PROJECT_SOURCE_DIR}/${python_src}
                       COMMAND ${CMAKE_COMMAND} -E copy
                       ${PROJECT_SOURCE_DIR}/${python_src}
                       ${CMAKE_BINARY_DIR}/${python_src})
    list(APPEND build_files ${CMAKE_BINARY_DIR}/${python_src})
  endforeach()

  add_custom_target(python_copy_files ALL DEPENDS ${build_files})


  # Install commands
  # Pick up static python files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe2 DESTINATION ${PYTHON_LIB_REL_PATH}
          FILES_MATCHING PATTERN "*.py")
  # Caffe proto files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe DESTINATION ${PYTHON_LIB_REL_PATH}
          FILES_MATCHING PATTERN "*.py")
  # Caffe2 proto files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe2 DESTINATION ${PYTHON_LIB_REL_PATH}
          FILES_MATCHING PATTERN "*.py")
endif()
