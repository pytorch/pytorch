# ---[ Generate and install header and cpp files
include(../cmake/Codegen.cmake)

# ---[ Vulkan code gen
if(USE_VULKAN)
  include(../cmake/VulkanCodegen.cmake)
endif()

# ---[ MSVC OpenMP modification
if(MSVC)
  include(../cmake/public/utils.cmake)
endif()

# ATen parallelism settings
#  OMP - OpenMP for intra-op, native thread pool for inter-op parallelism
#  NATIVE - using native thread pool for intra- and inter-op parallelism
#  TBB - using TBB for intra- and native thread pool for inter-op parallelism
if(INTERN_BUILD_MOBILE AND NOT BUILD_CAFFE2_MOBILE)
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
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/contrib/aten/aten_op.h
  COMMAND
  "${PYTHON_EXECUTABLE}" ${CMAKE_CURRENT_SOURCE_DIR}/contrib/aten/gen_op.py
    --aten_root=${CMAKE_CURRENT_SOURCE_DIR}/../aten
    --template_dir=${CMAKE_CURRENT_SOURCE_DIR}/contrib/aten
    --yaml_dir=${CMAKE_CURRENT_BINARY_DIR}/../aten/src/ATen
    --install_dir=${CMAKE_CURRENT_BINARY_DIR}/contrib/aten
  DEPENDS
  ATEN_CPU_FILES_GEN_TARGET
  ${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml
  ${CMAKE_CURRENT_SOURCE_DIR}/contrib/aten/gen_op.py
  ${CMAKE_CURRENT_SOURCE_DIR}/contrib/aten/aten_op_template.h)

  add_custom_target(__aten_op_header_gen
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/contrib/aten/aten_op.h)
  add_library(aten_op_header_gen INTERFACE)
  add_dependencies(aten_op_header_gen __aten_op_header_gen)

  # Add source, includes, and libs to lists
  list(APPEND Caffe2_CPU_SRCS ${ATen_CPU_SRCS})
  list(APPEND Caffe2_GPU_SRCS ${ATen_CUDA_SRCS})
  list(APPEND Caffe2_GPU_SRCS_W_SORT_BY_KEY ${ATen_CUDA_SRCS_W_SORT_BY_KEY})
  list(APPEND Caffe2_HIP_SRCS ${ATen_HIP_SRCS})
  list(APPEND Caffe2_HIP_SRCS ${ATen_HIP_SRCS_W_SORT_BY_KEY})
  list(APPEND Caffe2_CPU_TEST_SRCS ${ATen_CPU_TEST_SRCS})
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
if(BUILD_CAFFE2 AND (NOT INTERN_BUILD_MOBILE OR BUILD_CAFFE2_MOBILE))
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
      add_subdirectory(quantization)
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
if(NOT BUILD_CAFFE2)
  add_subdirectory(proto)
endif()

# Advanced: if we have allow list specified, we will do intersections for all
# main lib srcs.
if(CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_CPU_SRCS CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_GPU_SRCS CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_HIP_SRCS CAFFE2_ALLOWLISTED_FILES)
endif()

# Debug messages - if you want to get a list of source files, enable the
# following.
if(FALSE)
  message(STATUS "CPU sources: ")
  foreach(tmp ${Caffe2_CPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU sources: ")
  foreach(tmp ${Caffe2_GPU_SRCS})
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

  message(STATUS "HIP test sources: ")
  foreach(tmp ${Caffe2_HIP_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen CPU test sources: ")
  foreach(tmp ${ATen_CPU_TEST_SRCS})
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

if(NOT INTERN_BUILD_MOBILE OR BUILD_CAFFE2_MOBILE)
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



if(NOT INTERN_BUILD_MOBILE OR NOT BUILD_CAFFE2_MOBILE)
  if(USE_DISTRIBUTED)

    # Define this target even if we're building without TensorPipe, to make life
    # easier to other targets that depend on this. However, in that case, by not
    # setting the USE_TENSORPIPE compile definition, this target will just end
    # up being empty. Downstream targets should also add a #ifdef guard.
    if(NOT WIN32)
      add_library(process_group_agent "${TORCH_SRC_DIR}/csrc/distributed/rpc/process_group_agent.cpp" "${TORCH_SRC_DIR}/csrc/distributed/rpc/process_group_agent.h")
      target_link_libraries(process_group_agent PRIVATE torch c10d fmt::fmt-header-only)
      add_dependencies(process_group_agent torch c10d)

      add_library(tensorpipe_agent
        "${TORCH_SRC_DIR}/csrc/distributed/rpc/tensorpipe_agent.cpp"
        "${TORCH_SRC_DIR}/csrc/distributed/rpc/tensorpipe_agent.h"
        "${TORCH_SRC_DIR}/csrc/distributed/rpc/tensorpipe_utils.cpp"
        "${TORCH_SRC_DIR}/csrc/distributed/rpc/tensorpipe_utils.h"
        )
      target_link_libraries(tensorpipe_agent PRIVATE torch c10d tensorpipe fmt::fmt-header-only)
      add_dependencies(tensorpipe_agent torch c10d)
      if(USE_TENSORPIPE)
        if(USE_CUDA)
          target_compile_definitions(tensorpipe_agent PUBLIC USE_CUDA)
        endif()

        if(USE_ROCM)
          target_compile_definitions(tensorpipe_agent PRIVATE
            USE_ROCM
            __HIP_PLATFORM_HCC__
          )
        endif()

        target_compile_definitions(tensorpipe_agent PUBLIC USE_TENSORPIPE)
        target_link_libraries(tensorpipe_agent PRIVATE tensorpipe)
        add_dependencies(tensorpipe_agent tensorpipe)
      endif()
    endif()
  endif()

  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

  # Generate files
  set(TOOLS_PATH "${TORCH_ROOT}/tools")

  configure_file("${TORCH_SRC_DIR}/_utils_internal.py"
    "${TOOLS_PATH}/shared/_utils_internal.py"
    COPYONLY)


  set(GENERATED_CXX_TORCH
    "${TORCH_SRC_DIR}/csrc/autograd/generated/Functions.cpp"
    )

  if(NOT INTERN_DISABLE_AUTOGRAD)
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
    )
  endif()

  set(GENERATED_H_TORCH
    "${TORCH_SRC_DIR}/csrc/autograd/generated/Functions.h"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/variable_factories.h"
    )

  if(NOT INTERN_DISABLE_AUTOGRAD)
    list(APPEND GENERATED_H_TORCH
      "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType.h"
    )
  endif()

  set(GENERATED_CXX_PYTHON
    "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/python_variable_methods.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/python_torch_functions.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/python_nn_functions.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/python_fft_functions.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/python_linalg_functions.cpp"
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

  add_custom_command(
    OUTPUT
    ${TORCH_GENERATED_CODE}
    COMMAND
    "${PYTHON_EXECUTABLE}" tools/setup_helpers/generate_code.py
      --declarations-path "${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml"
      --native-functions-path "aten/src/ATen/native/native_functions.yaml"
      --nn-path "aten/src"
      $<$<BOOL:${INTERN_DISABLE_AUTOGRAD}>:--disable-autograd>
      $<$<BOOL:${SELECTED_OP_LIST}>:--selected-op-list-path="${SELECTED_OP_LIST}">
      --force_schema_registration
    DEPENDS
    "${TORCH_ROOT}/aten/src/ATen/native/native_functions.yaml"
    "${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml"
    "${TOOLS_PATH}/autograd/templates/VariableType.h"
    "${TOOLS_PATH}/autograd/templates/VariableType.cpp"
    "${TOOLS_PATH}/autograd/templates/TraceType.cpp"
    "${TOOLS_PATH}/autograd/templates/Functions.h"
    "${TOOLS_PATH}/autograd/templates/Functions.cpp"
    "${TOOLS_PATH}/autograd/templates/python_functions.h"
    "${TOOLS_PATH}/autograd/templates/python_functions.cpp"
    "${TOOLS_PATH}/autograd/templates/python_variable_methods.cpp"
    "${TOOLS_PATH}/autograd/templates/python_torch_functions.cpp"
    "${TOOLS_PATH}/autograd/templates/python_nn_functions.cpp"
    "${TOOLS_PATH}/autograd/templates/python_fft_functions.cpp"
    "${TOOLS_PATH}/autograd/templates/python_linalg_functions.cpp"
    "${TOOLS_PATH}/autograd/templates/variable_factories.h"
    "${TOOLS_PATH}/autograd/templates/annotated_fn_args.py"
    "${TOOLS_PATH}/autograd/deprecated.yaml"
    "${TOOLS_PATH}/autograd/derivatives.yaml"
    "${TOOLS_PATH}/autograd/gen_autograd_functions.py"
    "${TOOLS_PATH}/autograd/gen_autograd.py"
    "${TOOLS_PATH}/autograd/gen_python_functions.py"
    "${TOOLS_PATH}/autograd/gen_variable_factories.py"
    "${TOOLS_PATH}/autograd/gen_variable_type.py"
    "${TOOLS_PATH}/autograd/load_derivatives.py"
    "${TOOLS_PATH}/autograd/nested_dict.py"
    "${TOOLS_PATH}/autograd/utils.py"
    WORKING_DIRECTORY "${TORCH_ROOT}")


  # Required workaround for libtorch_python.so build
  # see https://samthursfield.wordpress.com/2015/11/21/cmake-dependencies-between-targets-and-files-and-custom-commands/#custom-commands-in-different-directories
  add_custom_target(
    generate-torch-sources
    DEPENDS ${TORCH_GENERATED_CODE}
    )

  set(TORCH_SRCS ${GENERATED_CXX_TORCH})
  list(APPEND TORCH_SRCS ${GENERATED_H_TORCH})
  append_filelist("libtorch_cmake_sources" TORCH_SRCS)

  # Required workaround for LLVM 9 includes.
  if(NOT MSVC)
    set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/tensorexpr/llvm_jit.cpp PROPERTIES COMPILE_FLAGS -Wno-noexcept-type)
  endif()
  # Disable certain warnings for GCC-9.X
  if(CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0.0))
    # See https://github.com/pytorch/pytorch/issues/38856
    set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/tensorexpr/llvm_jit.cpp PROPERTIES COMPILE_FLAGS "-Wno-redundant-move -Wno-noexcept-type")
    set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/tensorexpr/llvm_codegen.cpp PROPERTIES COMPILE_FLAGS -Wno-init-list-lifetime)
  endif()

  if(NOT INTERN_DISABLE_MOBILE_INTERP)
    set(MOBILE_SRCS
       ${TORCH_SRC_DIR}/csrc/jit/mobile/function.cpp
       ${TORCH_SRC_DIR}/csrc/jit/mobile/import.cpp
       ${TORCH_SRC_DIR}/csrc/jit/mobile/import_data.cpp
       ${TORCH_SRC_DIR}/csrc/jit/mobile/module.cpp
       ${TORCH_SRC_DIR}/csrc/jit/mobile/observer.cpp
       ${TORCH_SRC_DIR}/csrc/jit/mobile/interpreter.cpp
       ${TORCH_SRC_DIR}/csrc/jit/mobile/export_data.cpp
       ${TORCH_SRC_DIR}/csrc/jit/mobile/optim/sgd.cpp
       ${TORCH_SRC_DIR}/csrc/jit/mobile/sequential.cpp
       )
    list(APPEND TORCH_SRCS ${MOBILE_SRCS})
  endif()

  # This one needs to be unconditionally added as Functions.cpp is also unconditionally added
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/autograd/FunctionsManual.cpp
    ${TORCH_SRC_DIR}/csrc/utils/out_types.cpp
  )

  if(NOT INTERN_DISABLE_AUTOGRAD)
    list(APPEND TORCH_SRCS
      ${TORCH_SRC_DIR}/csrc/autograd/TraceTypeManual.cpp
      ${TORCH_SRC_DIR}/csrc/autograd/VariableTypeManual.cpp
    )
  endif()

  if(NOT INTERN_BUILD_MOBILE)
    list(APPEND TORCH_SRCS
      ${TORCH_SRC_DIR}/csrc/api/src/jit.cpp
      ${TORCH_SRC_DIR}/csrc/jit/serialization/onnx.cpp
      ${TORCH_SRC_DIR}/csrc/jit/serialization/export.cpp
      ${TORCH_SRC_DIR}/csrc/jit/serialization/export_module.cpp
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
    if(USE_DISTRIBUTED AND NOT WIN32)
      append_filelist("libtorch_distributed_sources" TORCH_SRCS)
    endif()
  endif()

  if(USE_CUDA OR USE_ROCM)
    append_filelist("libtorch_cuda_core_sources" Caffe2_GPU_HIP_JIT_FUSERS_SRCS)
  endif()

  if(USE_CUDA)
    list(APPEND Caffe2_GPU_SRCS ${Caffe2_GPU_HIP_JIT_FUSERS_SRCS})
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
  endif()

  if(USE_ROCM)
    list(APPEND Caffe2_HIP_SRCS ${Caffe2_GPU_HIP_JIT_FUSERS_SRCS})
    if(USE_NCCL)
      list(APPEND Caffe2_HIP_SRCS
        ${TORCH_SRC_DIR}/csrc/cuda/nccl.cpp)
    endif()
    # caffe2_nvrtc's stubs to driver APIs are useful for HIP.
    # See NOTE [ ATen NVRTC Stub and HIP ]
    add_library(caffe2_nvrtc SHARED ${ATen_NVRTC_STUB_SRCS})
    target_link_libraries(caffe2_nvrtc ${PYTORCH_HIP_HCC_LIBRARIES} ${ROCM_HIPRTC_LIB})
    target_compile_definitions(caffe2_nvrtc PRIVATE USE_ROCM __HIP_PLATFORM_HCC__)
    install(TARGETS caffe2_nvrtc DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  endif()

  if(NOT NO_API)
    list(APPEND TORCH_SRCS
      ${TORCH_SRC_DIR}/csrc/api/src/cuda.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/data/datasets/mnist.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/distributed.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/random.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/sequential.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/stream.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/enum.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/serialize.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/jit.cpp
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
      ${TORCH_SRC_DIR}/csrc/api/src/serialize/input-archive.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/serialize/output-archive.cpp
    )
  endif()

  list(APPEND Caffe2_CPU_SRCS ${TORCH_SRCS})
endif()

# NOTE [ Linking AVX and non-AVX files ]
#
# Regardless of the CPU capabilities, we build some files with AVX and AVX2
# instruction set. If the host CPU doesn't support those, we simply ignore their
# functions at runtime during dispatch.
#
# We must make sure that those files are at the end of the input list when
# linking the torch_cpu library. Otherwise, the following error scenario might
# occur:
# 1. A non-AVX and an AVX file both call a function defined with the `inline`
#    keyword
# 2. The compiler decides not to inline this function
# 3. Two different versions of the machine code are generated for this function:
#    one without AVX instructions and one with AVX.
# 4. When linking, the AVX version is found earlier in the input object files,
#    so the linker makes the entire library use it, even in code not guarded by
#    the dispatcher.
# 5. A CPU without AVX support executes this function, encounters an AVX
#    instruction and crashes.
#
# Thus we organize the input files in the following order:
# 1. All files with no AVX support
# 2. All files with AVX support (conveniently, they all have names ending with
#    'AVX.cpp')
# 3. All files with AVX2 support ('*AVX2.cpp')
set(Caffe2_CPU_SRCS_NON_AVX)
set(Caffe2_CPU_SRCS_AVX)
set(Caffe2_CPU_SRCS_AVX2)
foreach(input_filename ${Caffe2_CPU_SRCS})
  if(${input_filename} MATCHES "AVX\\.cpp")
    list(APPEND Caffe2_CPU_SRCS_AVX ${input_filename})
  elseif(${input_filename} MATCHES "AVX2\\.cpp")
    list(APPEND Caffe2_CPU_SRCS_AVX2 ${input_filename})
  else()
    list(APPEND Caffe2_CPU_SRCS_NON_AVX ${input_filename})
  endif()
endforeach(input_filename)
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS_NON_AVX} ${Caffe2_CPU_SRCS_AVX} ${Caffe2_CPU_SRCS_AVX2})

# ==========================================================
# END formerly-libtorch sources
# ==========================================================

add_library(torch_cpu ${Caffe2_CPU_SRCS})
if(HAVE_SOVERSION)
  set_target_properties(torch_cpu PROPERTIES
      VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
endif()
torch_compile_options(torch_cpu)  # see cmake/public/utils.cmake

if(USE_LLVM AND LLVM_FOUND)
  llvm_map_components_to_libnames(LLVM_LINK_LIBS
    support core analysis executionengine instcombine
    scalaropts transformutils native orcjit)
  target_link_libraries(torch_cpu PRIVATE ${LLVM_LINK_LIBS})
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
elseif(USE_CUDA)
  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
  if(CUDA_SEPARABLE_COMPILATION)
    # Separate compilation fails when kernels using `thrust::sort_by_key`
    # are linked with the rest of CUDA code. Workaround by linking the separateley
    set(_generated_name "torch_cuda_w_sort_by_key_intermediate_link${CMAKE_C_OUTPUT_EXTENSION}")
    set(torch_cuda_w_sort_by_key_link_file "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/torch_cuda.dir/${CMAKE_CFG_INTDIR}/${_generated_name}")
    cuda_wrap_srcs(torch_cuda OBJ Caffe2_GPU_W_SORT_BY_KEY_OBJ ${Caffe2_GPU_SRCS_W_SORT_BY_KEY})
    CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS("${torch_cuda_w_sort_by_key_link_file}" torch_cpu "${_options}" "${torch_cuda_SEPARABLE_COMPILATION_OBJECTS}")
    set( torch_cuda_SEPARABLE_COMPILATION_OBJECTS )
    # Pass compiled sort-by-key object + device-linked fatbin as extra dependencies of torch_cuda
    cuda_add_library(torch_cuda ${Caffe2_GPU_SRCS} ${torch_cuda_w_sort_by_key_link_file} ${Caffe2_GPU_W_SORT_BY_KEY_OBJ})
  else()
    cuda_add_library(torch_cuda ${Caffe2_GPU_SRCS} ${Caffe2_GPU_SRCS_W_SORT_BY_KEY})
  endif()
  set(CUDA_LINK_LIBRARIES_KEYWORD)
  torch_compile_options(torch_cuda)  # see cmake/public/utils.cmake
  if(USE_NCCL)
    target_link_libraries(torch_cuda PRIVATE __caffe2_nccl)
    target_compile_definitions(torch_cuda PRIVATE USE_NCCL)
  endif()
endif()

if(USE_CUDA OR USE_ROCM)
  if(USE_CUDA)
    set(TORCHLIB_FLAVOR torch_cuda)
  elseif(USE_ROCM)
    set(TORCHLIB_FLAVOR torch_hip)
  endif()

  # The list of NVFUSER runtime files
  list(APPEND NVFUSER_RUNTIME_FILES
    ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/block_reduction.cu
    ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/broadcast.cu
    ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/fp16_support.cu
    ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/grid_reduction.cu
    ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/helpers.cu
    ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/random_numbers.cu
    ${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/runtime/tensor.cu
  )

  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/nvfuser_resources")

  # "stringify" NVFUSER runtime sources
  # (generate C++ header files embedding the original input as a string literal)
  set(NVFUSER_STRINGIFY_TOOL "${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/tools/stringify_file.py")
  foreach(src ${NVFUSER_RUNTIME_FILES})
    get_filename_component(filename ${src} NAME_WE)
    set(dst "${CMAKE_BINARY_DIR}/include/nvfuser_resources/${filename}.h")
    add_custom_command(
      COMMENT "Stringify NVFUSER runtime source file"
      OUTPUT ${dst}
      DEPENDS ${src}
      COMMAND ${PYTHON_EXECUTABLE} ${NVFUSER_STRINGIFY_TOOL} -i ${src} -o ${dst}
    )
    add_custom_target(nvfuser_rt_${filename} DEPENDS ${dst})
    add_dependencies(${TORCHLIB_FLAVOR} nvfuser_rt_${filename})

    # also generate the resource headers during the configuration step
    # (so tools like clang-tidy can run w/o requiring a real build)
    execute_process(COMMAND
      ${PYTHON_EXECUTABLE} ${NVFUSER_STRINGIFY_TOOL} -i ${src} -o ${dst})
  endforeach()

  target_include_directories(${TORCHLIB_FLAVOR} PRIVATE "${CMAKE_BINARY_DIR}/include")
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

if(NOT INTERN_BUILD_MOBILE OR NOT BUILD_CAFFE2_MOBILE)
  if(NOT NO_API)
    target_include_directories(torch_cpu PRIVATE
      ${TORCH_SRC_DIR}/csrc/api
      ${TORCH_SRC_DIR}/csrc/api/include)
  endif()

  if(USE_CUDA AND MSVC)
    # -INCLUDE is used to ensure torch_cuda is linked against in a project that relies on it.
    # Related issue: https://github.com/pytorch/pytorch/issues/31611
    target_link_libraries(torch_cuda INTERFACE "-INCLUDE:?warp_size@cuda@at@@YAHXZ")
  endif()


  set(TH_CPU_INCLUDE
    # dense
    aten/src/TH
    ${CMAKE_CURRENT_BINARY_DIR}/aten/src/TH
    ${TORCH_ROOT}/aten/src
    ${CMAKE_CURRENT_BINARY_DIR}/aten/src
    ${CMAKE_BINARY_DIR}/aten/src)
  target_include_directories(torch_cpu PRIVATE ${TH_CPU_INCLUDE})

  set(ATen_CPU_INCLUDE
    ${TORCH_ROOT}/aten/src
    ${CMAKE_CURRENT_BINARY_DIR}/../aten/src
    ${CMAKE_CURRENT_BINARY_DIR}/../aten/src/ATen
    ${CMAKE_BINARY_DIR}/aten/src)

if(USE_TBB)
  list(APPEND ATen_CPU_INCLUDE ${TBB_ROOT_DIR}/include)
  target_link_libraries(torch_cpu PUBLIC tbb)
endif()


  target_include_directories(torch_cpu PRIVATE ${ATen_CPU_INCLUDE})

  target_include_directories(torch_cpu PRIVATE
    ${TORCH_SRC_DIR}/csrc)

  target_include_directories(torch_cpu PRIVATE
    ${TORCH_ROOT}/third_party/miniz-2.0.8)

  if(USE_KINETO)
    target_include_directories(torch_cpu PRIVATE
      ${TORCH_ROOT}/third_party/kineto/libkineto/include
      ${TORCH_ROOT}/third_party/kineto/libkineto/src)
  endif()

  install(DIRECTORY "${TORCH_SRC_DIR}/csrc"
    DESTINATION ${TORCH_INSTALL_INCLUDE_DIR}/torch
    FILES_MATCHING PATTERN "*.h")
  install(FILES
    "${TORCH_SRC_DIR}/script.h"
    "${TORCH_SRC_DIR}/extension.h"
    "${TORCH_SRC_DIR}/custom_class.h"
    "${TORCH_SRC_DIR}/library.h"
    "${TORCH_SRC_DIR}/custom_class_detail.h"
    DESTINATION ${TORCH_INSTALL_INCLUDE_DIR}/torch)


  if(BUILD_TEST)
    add_subdirectory(${TORCH_ROOT}/test/cpp/jit ${CMAKE_BINARY_DIR}/test_jit)
    add_subdirectory(${TORCH_ROOT}/test/cpp/tensorexpr ${CMAKE_BINARY_DIR}/test_tensorexpr)
    if(USE_DISTRIBUTED AND NOT WIN32)
      add_subdirectory(${TORCH_ROOT}/test/cpp/rpc ${CMAKE_BINARY_DIR}/test_cpp_rpc)
    endif()
  endif()

  if(BUILD_TEST AND NOT NO_API)
    add_subdirectory(${TORCH_ROOT}/test/cpp/api ${CMAKE_BINARY_DIR}/test_api)
    add_subdirectory(${TORCH_ROOT}/test/cpp/dist_autograd ${CMAKE_BINARY_DIR}/dist_autograd)
  endif()

  # XXX This ABI check cannot be run with arm-linux-androideabi-g++
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if(DEFINED GLIBCXX_USE_CXX11_ABI)
      message(STATUS "_GLIBCXX_USE_CXX11_ABI is already defined as a cmake variable")
    else()
      message(STATUS "${CMAKE_CXX_COMPILER} ${TORCH_SRC_DIR}/abi-check.cpp -o ${CMAKE_BINARY_DIR}/abi-check")
      execute_process(
        COMMAND
        "${CMAKE_CXX_COMPILER}"
        "${TORCH_SRC_DIR}/abi-check.cpp"
        "-o"
        "${CMAKE_BINARY_DIR}/abi-check"
        RESULT_VARIABLE ABI_CHECK_COMPILE_RESULT)
      if(ABI_CHECK_COMPILE_RESULT)
        message(FATAL_ERROR "Could not compile ABI Check: ${ABI_CHECK_COMPILE_RESULT}")
      endif()
      execute_process(
        COMMAND "${CMAKE_BINARY_DIR}/abi-check"
        RESULT_VARIABLE ABI_CHECK_RESULT
        OUTPUT_VARIABLE GLIBCXX_USE_CXX11_ABI)
      if(ABI_CHECK_RESULT)
        message(WARNING "Could not run ABI Check: ${ABI_CHECK_RESULT}")
      endif()
    endif()
    message(STATUS "Determined _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
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

  if(USE_DISTRIBUTED)
    add_subdirectory(${TORCH_SRC_DIR}/lib/c10d lib_c10d)
  endif()


  # ---[ Torch python bindings build
  add_subdirectory(../torch torch)


endif()
# ==========================================================
# END formerly-libtorch flags
# ==========================================================






if(NOT NO_API)
  target_include_directories(torch_cpu PUBLIC
    $<BUILD_INTERFACE:${TORCH_SRC_DIR}/csrc/api>
    $<BUILD_INTERFACE:${TORCH_SRC_DIR}/csrc/api/include>)
endif()


if(USE_OPENMP)
  find_package(OpenMP QUIET)
endif()
if(USE_OPENMP AND OPENMP_FOUND)
  if(MSVC AND OpenMP_CXX_LIBRARIES MATCHES "libiomp5md\\.lib")
    set(AT_MKL_MT 1)
  else()
    set(AT_MKL_MT 0)
  endif()
  message(STATUS "pytorch is compiling with OpenMP. \n"
    "OpenMP CXX_FLAGS: ${OpenMP_CXX_FLAGS}. \n"
    "OpenMP libraries: ${OpenMP_CXX_LIBRARIES}.")
  target_compile_options(torch_cpu PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(torch_cpu PRIVATE ${OpenMP_CXX_LIBRARIES})
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

# Pass USE_DISTRIBUTED to torch_cpu, as some codes in jit/pickler.cpp and
# jit/unpickler.cpp need to be compiled only when USE_DISTRIBUTED is set
if(USE_DISTRIBUTED)
  target_compile_definitions(torch_cpu PRIVATE
    USE_DISTRIBUTED
  )
  # Pass USE_RPC in order to reduce use of
  # #if defined(USE_DISTRIBUTED) && !defined(_WIN32)
  # need to be removed when RPC is supported
  if(NOT WIN32)
    target_compile_definitions(torch_cpu PRIVATE
      USE_RPC
    )
  endif()
  # Pass USE_TENSORPIPE to torch_cpu as some parts of rpc/utils.cpp
  # can only be compiled with USE_TENSORPIPE is set.
  if(USE_TENSORPIPE)
    target_compile_definitions(torch_cpu PRIVATE
      USE_TENSORPIPE
    )
  endif()
endif()

if(NOT INTERN_BUILD_MOBILE OR BUILD_CAFFE2_MOBILE)
  caffe2_interface_library(caffe2_protos caffe2_protos_whole)
  target_link_libraries(torch_cpu PRIVATE caffe2_protos_whole)
  if(${CAFFE2_LINK_LOCAL_PROTOBUF})
    target_link_libraries(torch_cpu INTERFACE protobuf::libprotobuf)
  else()
    target_link_libraries(torch_cpu PUBLIC protobuf::libprotobuf)
  endif()
endif()

if(USE_OPENMP AND OPENMP_FOUND)
  message(STATUS "Caffe2 is compiling with OpenMP. \n"
    "OpenMP CXX_FLAGS: ${OpenMP_CXX_FLAGS}. \n"
    "OpenMP libraries: ${OpenMP_CXX_LIBRARIES}.")
  target_link_libraries(torch_cpu PRIVATE ${OpenMP_CXX_LIBRARIES})
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
# Set standard properties on the target
torch_set_target_props(torch_cpu)


target_compile_options(torch_cpu PRIVATE "-DCAFFE2_BUILD_MAIN_LIB")
if(USE_CUDA)
  target_compile_options(torch_cuda PRIVATE "-DTORCH_CUDA_BUILD_MAIN_LIB")
  # NB: This must be target_compile_definitions, not target_compile_options,
  # as the latter is not respected by nvcc
  target_compile_definitions(torch_cuda PRIVATE "-DTORCH_CUDA_BUILD_MAIN_LIB")
elseif(USE_ROCM)
  target_compile_options(torch_hip PRIVATE "-DTORCH_HIP_BUILD_MAIN_LIB")
  target_compile_definitions(torch_hip PRIVATE "-DTORCH_HIP_BUILD_MAIN_LIB")
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

  install(TARGETS torch_global_deps DESTINATION "${TORCH_INSTALL_LIB_DIR}")
endif()

# ---[ Caffe2 HIP sources.
if(USE_ROCM)
  # Call again since Caffe2_HIP_INCLUDE is extended with ATen include dirs.
  # Get Compile Definitions from the directory (FindHIP.cmake bug)
  get_directory_property(MY_DEFINITIONS COMPILE_DEFINITIONS)
  if(MY_DEFINITIONS)
    foreach(_item ${MY_DEFINITIONS})
      list(APPEND HIP_HCC_FLAGS "-D${_item}")
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
  target_link_libraries(torch_hip PUBLIC torch_cpu_library ${Caffe2_HIP_DEPENDENCY_LIBS})

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

  foreach(test_src ${ATen_VEC256_TEST_SRCS})
    foreach(i RANGE ${NUM_CPU_CAPABILITY_NAMES})
        get_filename_component(test_name ${test_src} NAME_WE)
        list(GET CPU_CAPABILITY_NAMES ${i} CPU_CAPABILITY)
        list(GET CPU_CAPABILITY_FLAGS ${i} FLAGS)
        separate_arguments(FLAGS UNIX_COMMAND "${FLAGS}")
        add_executable(${test_name}_${CPU_CAPABILITY} "${test_src}")
        target_link_libraries(${test_name}_${CPU_CAPABILITY} torch_library gtest_main)
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
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    if(INSTALL_TEST)
      install(TARGETS ${test_name} DESTINATION test)
      # Install PDB files for MSVC builds
      if(MSVC AND BUILD_SHARED_LIBS)
        install(FILES $<TARGET_PDB_FILE:${test_name}> DESTINATION test OPTIONAL)
      endif()
    endif()
  endforeach()

  if(USE_CUDA)
    foreach(test_src ${Caffe2_GPU_TEST_SRCS})
      get_filename_component(test_name ${test_src} NAME_WE)
      cuda_add_executable(${test_name} "${test_src}")
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
  if(USE_MPI)
    target_link_libraries(mpi_test ${MPI_CXX_LIBRARIES})
    if(USE_CUDA)
      target_link_libraries(mpi_gpu_test ${MPI_CXX_LIBRARIES})
    endif()
  endif()
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
  # https://github.com/pytorch/pytorch/tree/master/tools/build_pytorch_libs.bat#note-backslash-munging-on-windows
  pycmd(PYTHON_SITE_PACKAGES "
      import os
      from distutils import sysconfig
      print(sysconfig.get_python_lib(prefix=''))
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
      from distutils import sysconfig
      ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
      print(ext_suffix if ext_suffix else '')
  ")
  if("${PY_EXT_SUFFIX}" STREQUAL "")
    if(MSVC)
      set(PY_EXT_SUFFIX ".pyd")
    else()
      set(PY_EXT_SUFFIX ".so")
    endif()
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
  if(NOT MSVC)
    set_target_properties(caffe2_pybind11_state PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  endif()
  torch_set_target_props(caffe2_pybind11_state)
  set_target_properties(caffe2_pybind11_state PROPERTIES PREFIX "" DEBUG_POSTFIX "")
  set_target_properties(caffe2_pybind11_state PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
  set_target_properties(caffe2_pybind11_state PROPERTIES LINK_FLAGS "${_caffe2_pybind11_state_linker_flags}")
  target_include_directories(caffe2_pybind11_state PRIVATE $<INSTALL_INTERFACE:include>)
  target_include_directories(caffe2_pybind11_state PRIVATE ${Caffe2_CPU_INCLUDE})

  target_link_libraries(
      caffe2_pybind11_state torch_library)
  if(WIN32)
    target_link_libraries(caffe2_pybind11_state ${PYTHON_LIBRARIES})
    target_link_libraries(caffe2_pybind11_state onnx_proto)
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
    if(NOT MSVC)
      set_target_properties(caffe2_pybind11_state_gpu PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
    endif()
    torch_set_target_props(caffe2_pybind11_state_gpu)
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES PREFIX "" DEBUG_POSTFIX "")
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES LINK_FLAGS "${_caffe2_pybind11_state_linker_flags}")
    target_include_directories(caffe2_pybind11_state_gpu PRIVATE $<INSTALL_INTERFACE:include>)
    target_include_directories(caffe2_pybind11_state_gpu PRIVATE ${Caffe2_CPU_INCLUDE})
    target_link_libraries(caffe2_pybind11_state_gpu torch_library)
    if(WIN32)
      target_link_libraries(caffe2_pybind11_state_gpu ${PYTHON_LIBRARIES})
      target_link_libraries(caffe2_pybind11_state_gpu onnx_proto)
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
    if(NOT MSVC)
      target_compile_options(caffe2_pybind11_state_hip PRIVATE ${HIP_CXX_FLAGS} -fvisibility=hidden)
    endif()
    torch_set_target_props(caffe2_pybind11_state_hip)
    set_target_properties(caffe2_pybind11_state_hip PROPERTIES PREFIX "")
    set_target_properties(caffe2_pybind11_state_hip PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
    set_target_properties(caffe2_pybind11_state_hip PROPERTIES LINK_FLAGS "${_caffe2_pybind11_state_linker_flags}")
    target_include_directories(caffe2_pybind11_state_hip PRIVATE $<INSTALL_INTERFACE:include>)
    target_include_directories(caffe2_pybind11_state_hip PRIVATE ${Caffe2_CPU_INCLUDE} ${Caffe2_HIP_INCLUDE})
    target_link_libraries(caffe2_pybind11_state_hip torch_library)
    if(WIN32)
      target_link_libraries(caffe2_pybind11_state_hip ${PYTHON_LIBRARIES})
    endif(WIN32)

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
