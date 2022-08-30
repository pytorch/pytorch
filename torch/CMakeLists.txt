# This file used to build libtorch.so.
# Now it only builds the Torch python bindings.

if(NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
  project(torch CXX C)
  find_package(torch REQUIRED)
  option(USE_CUDA "Use CUDA" ON)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(NOT BUILD_PYTHON)
  return()
endif()

if(USE_TBB)
  include_directories(${TBB_INCLUDE_DIR})
endif()

set(TORCH_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TORCH_ROOT "${TORCH_SRC_DIR}/..")

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()

if(MSVC)
    set(LIBSHM_SUBDIR libshm_windows)
else()
    set(LIBSHM_SUBDIR libshm)
endif()

set(LIBSHM_SRCDIR ${TORCH_SRC_DIR}/lib/${LIBSHM_SUBDIR})
add_subdirectory(${LIBSHM_SRCDIR})


# Generate files
set(TOOLS_PATH "${TORCH_ROOT}/tools")


set(TORCH_PYTHON_SRCS
    ${GENERATED_THNN_CXX}
    ${GENERATED_CXX_PYTHON}
    )
append_filelist("libtorch_python_core_sources" TORCH_PYTHON_SRCS)

list(APPEND TORCH_PYTHON_SRCS
    ${TORCH_SRC_DIR}/csrc/init_flatbuffer_module.cpp)

# NB: This has to match the condition under which the JIT test directory
#     is included (at the time of writing that's in caffe2/CMakeLists.txt).
if(BUILD_TEST)
    add_definitions(-DBUILDING_TESTS)
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_ROOT}/test/cpp/jit/torch_python_test.cpp
      )
endif()

set(TORCH_PYTHON_INCLUDE_DIRECTORIES
    ${PYTHON_INCLUDE_DIR}

    ${TORCH_ROOT}
    ${TORCH_ROOT}/aten/src
    ${TORCH_ROOT}/aten/src/TH

    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/aten/src
    ${CMAKE_BINARY_DIR}/caffe2/aten/src
    ${CMAKE_BINARY_DIR}/third_party
    ${CMAKE_BINARY_DIR}/third_party/onnx

    ${TORCH_ROOT}/third_party/valgrind-headers

    ${TORCH_ROOT}/third_party/gloo
    ${TORCH_ROOT}/third_party/onnx
    ${TORCH_ROOT}/third_party/flatbuffers/include
    ${TORCH_ROOT}/third_party/kineto/libkineto/include

    ${TORCH_SRC_DIR}/csrc
    ${TORCH_SRC_DIR}/csrc/api/include
    ${TORCH_SRC_DIR}/lib
    )

list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES ${LIBSHM_SRCDIR})

set(TORCH_PYTHON_LINK_LIBRARIES
    python::python
    pybind::pybind11
    shm
    fmt::fmt-header-only
    ATEN_CPU_FILES_GEN_LIB)

set(TORCH_PYTHON_COMPILE_DEFINITIONS)

set(TORCH_PYTHON_PUBLIC_COMPILE_DEFINITIONS)

set(TORCH_PYTHON_COMPILE_OPTIONS)

set(TORCH_PYTHON_LINK_FLAGS "")

if(MSVC)
    string(APPEND TORCH_PYTHON_LINK_FLAGS " /NODEFAULTLIB:LIBCMT.LIB")
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${PYTHON_LIBRARIES} onnx_library)
    if(NOT CMAKE_BUILD_TYPE MATCHES "Release")
      string(APPEND TORCH_PYTHON_LINK_FLAGS " /DEBUG:FULL")
    endif()
elseif(APPLE)
    string(APPEND TORCH_PYTHON_LINK_FLAGS " -undefined dynamic_lookup")
else()
    list(APPEND TORCH_PYTHON_COMPILE_OPTIONS
      -fno-strict-aliasing
      -Wno-write-strings
      -Wno-strict-aliasing)
endif()

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    list(APPEND TORCH_PYTHON_COMPILE_OPTIONS
      -Wno-writable-strings)
endif()

if(USE_ITT)
  list(APPEND TORCH_PYTHON_SRCS
    ${TORCH_SRC_DIR}/csrc/itt.cpp
  )
  list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_ITT)
endif()

if(USE_CUDA)
    include(${TORCH_ROOT}/cmake/public/cuda.cmake)
    append_filelist("libtorch_python_cuda_core_sources" TORCH_PYTHON_SRCS)
    list(APPEND TORCH_PYTHON_SRCS ${GENERATED_THNN_CXX_CUDA})

    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUDA)
    if(USE_CUDNN)
        list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUDNN)
    endif()

    list(APPEND TORCH_PYTHON_LINK_LIBRARIES torch::nvtoolsext)
endif()

if(USE_ROCM)
    append_filelist("libtorch_python_cuda_core_sources" TORCH_PYTHON_SRCS)
    list(APPEND TORCH_PYTHON_SRCS ${GENERATED_THNN_CXX_CUDA})

    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS
      USE_ROCM
      __HIP_PLATFORM_HCC__
      )
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${ROCM_ROCTX_LIB})
    list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES ${roctracer_INCLUDE_DIRS})
endif()

if(USE_EXPERIMENTAL_CUDNN_V8_API)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_EXPERIMENTAL_CUDNN_V8_API)
endif()

if(USE_CUDNN OR USE_ROCM)
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/shared/cudnn.cpp
      )
    if(USE_STATIC_CUDNN)
        set_source_files_properties(
          ${TORCH_SRC_DIR}/csrc/cuda/shared/cudnn.cpp
          PROPERTIES COMPILE_DEFINITIONS "USE_STATIC_CUDNN"
        )
    endif()
endif()

if(USE_MPS)
    list(APPEND TORCH_PYTHON_SRCS ${MPS_PYTHON_SRCS})
endif()

if(USE_VALGRIND AND NOT WIN32)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_VALGRIND)
endif()

# In the most recent CMake versions, a new 'TRANSFORM' subcommand of 'list' allows much of the boilerplate of defining the lists
# of type stub files to be omitted.
# For compatibility with older CMake versions, we omit it for now, but leave it as a comment in case compatibility with the older
# CMake versions is eventually dropped.
# set(Modules
#     __init__
#     activation
#     adaptive
#     batchnorm
#     container
#     conv
#     distance
#     dropout
#     fold
#     instancenorm
#     linear
#     loss
#     module
#     normalization
#     padding
#     pixelshuffle
#     pooling
#     rnn
#     sparse
#     upsampling
# )
# list(TRANSFORM Modules PREPEND "${TORCH_SRC_DIR}/nn/modules/")
add_custom_target(torch_python_stubs DEPENDS
    "${TORCH_SRC_DIR}/_C/__init__.pyi"
    "${TORCH_SRC_DIR}/_C/_VariableFunctions.pyi"
    "${TORCH_SRC_DIR}/nn/functional.pyi"
    "${TORCH_SRC_DIR}/utils/data/datapipes/datapipe.pyi"
)

file(GLOB_RECURSE torchgen_python "${PROJECT_SOURCE_DIR}/torchgen/*.py")
file(GLOB_RECURSE autograd_python "${TOOLS_PATH}/autograd/*.py")
file(GLOB_RECURSE pyi_python "${TOOLS_PATH}/pyi/*.py")
add_custom_command(
    OUTPUT
    "${TORCH_SRC_DIR}/_C/__init__.pyi"
    "${TORCH_SRC_DIR}/_C/_VariableFunctions.pyi"
    "${TORCH_SRC_DIR}/nn/functional.pyi"
    COMMAND
    "${PYTHON_EXECUTABLE}" -mtools.pyi.gen_pyi
      --native-functions-path "aten/src/ATen/native/native_functions.yaml"
      --tags-path "aten/src/ATen/native/tags.yaml"
      --deprecated-functions-path "tools/autograd/deprecated.yaml"
    DEPENDS
      "${TORCH_SRC_DIR}/_C/__init__.pyi.in"
      "${TORCH_SRC_DIR}/_C/_VariableFunctions.pyi.in"
      "${TORCH_SRC_DIR}/nn/functional.pyi.in"
      "${TORCH_ROOT}/aten/src/ATen/native/native_functions.yaml"
      "${TORCH_ROOT}/aten/src/ATen/native/tags.yaml"
      "${TORCH_ROOT}/tools/autograd/deprecated.yaml"
      ${pyi_python}
      ${autograd_python}
      ${torchgen_python}
    WORKING_DIRECTORY
    "${TORCH_ROOT}"
)
file(GLOB_RECURSE datapipe_files "${TORCH_SRC_DIR}/utils/data/datapipes/*.py")
add_custom_command(
    OUTPUT
    "${TORCH_SRC_DIR}/utils/data/datapipes/datapipe.pyi"
    COMMAND
    "${PYTHON_EXECUTABLE}" ${TORCH_SRC_DIR}/utils/data/datapipes/gen_pyi.py
    DEPENDS
    "${TORCH_SRC_DIR}/utils/data/datapipes/datapipe.pyi.in"
    ${datapipe_files}
    WORKING_DIRECTORY
    "${TORCH_ROOT}"
)
if(USE_DISTRIBUTED)
    if(WIN32)
      append_filelist("libtorch_python_distributed_core_sources" TORCH_PYTHON_SRCS)
    else()
      append_filelist("libtorch_python_distributed_sources" TORCH_PYTHON_SRCS)
    endif()
    # Disable certain warnings for GCC-9.X
    if(CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0.0))
      set_source_files_properties(${TORCH_SRC_DIR}/csrc/distributed/autograd/init.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
      set_source_files_properties(${TORCH_SRC_DIR}/csrc/distributed/rpc/testing/init.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
      set_source_files_properties(${TORCH_SRC_DIR}/csrc/distributed/c10d/init.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
    endif()
    # NCCL is a private dependency of libtorch, but libtorch_python includes
    # some private headers of libtorch, which in turn include NCCL. As a hacky
    # alternative to making NCCL a public dependency of libtorch, we make it
    # a private dependency of libtorch_python as well.
    if(USE_NCCL)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES __caffe2_nccl)
    endif()
    # Same for MPI.
    if(USE_MPI)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${MPI_CXX_LIBRARIES})
    endif()
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_C10D)

endif()

if(USE_NCCL AND NOT WIN32)
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/python_nccl.cpp)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_NCCL)
endif()


# WARNING- any TORCH_PYTHON_COMPILE_DEFINITIONS above this line
#          affect both torch_python and DEPLOY interpreter.
if(USE_DEPLOY)
  add_library(torch_python_obj OBJECT ${TORCH_PYTHON_SRCS})
  if(NOT MSVC)
    target_compile_options(torch_python_obj PRIVATE -Wno-unused-variable)
  endif()
  if(USE_DISTRIBUTED)
    # Set c10d-related compile definitions. For a "normal" build of
    # libtorch_python, these are set on libtorch as PUBLIC so they are
    # automatically propagated when libtorch_python links against libtorch. But
    # since in the deploy build we are intentionally *not* linking against
    # libtorch, we need to set them manually here.
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_DISTRIBUTED)
    if(USE_GLOO AND USE_C10D_GLOO)
      list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_C10D_GLOO)
    endif()
    if(USE_UCC AND USE_C10D_UCC)
      list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_C10D_UCC)
    endif()
    if(USE_NCCL AND USE_C10D_NCCL)
        list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_C10D_NCCL)
        # Put nccl headers on the include path. We are specifically only setting
        # include dirs here instead of linking against __caffe2_nccl wholesale
        # to ensure we aren't accidentally replicating the nccl lib.
        target_include_directories(torch_python_obj PRIVATE $<TARGET_PROPERTY:__caffe2_nccl,INTERFACE_INCLUDE_DIRECTORIES>)
    endif()
    if(USE_MPI AND USE_C10D_MPI)
      list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_C10D_MPI)
    endif()

    # Pass USE_RPC in order to reduce use of
    # #if defined(USE_DISTRIBUTED) && !defined(_WIN32)
    # need to be removed when RPC is supported
    if(NOT WIN32)
      target_compile_definitions(torch_cpu PUBLIC USE_RPC)
    endif()
    if(USE_TENSORPIPE)
      list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_TENSORPIPE)
    endif()

    # Set c10d-related include directories as well.
    target_include_directories(torch_python_obj PRIVATE $<BUILD_INTERFACE:${TORCH_SRC_DIR}/csrc/distributed>)
  endif()
  target_compile_definitions(torch_python_obj PRIVATE "-DTHP_BUILD_MAIN_LIB -DUSE_DEPLOY")

  target_compile_definitions(torch_python_obj PRIVATE ${TORCH_PYTHON_COMPILE_DEFINITIONS})

  target_compile_definitions(torch_python_obj PUBLIC ${TORCH_PYTHON_PUBLIC_COMPILE_DEFINITIONS})

  target_compile_options(torch_python_obj PRIVATE ${TORCH_PYTHON_COMPILE_OPTIONS})

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(torch_python_obj PRIVATE -fno-gnu-unique)
  endif()

  target_include_directories(torch_python_obj PUBLIC ${TORCH_PYTHON_INCLUDE_DIRECTORIES})
  target_include_directories(torch_python_obj PRIVATE ../third_party/fmt/include)

  # need to specify the dependency so the generated headers exist,
  # missing dependency since torch_python_obj doesn't link onnx, the interpreter lib does.
  add_dependencies(torch_python_obj onnx)

  target_include_directories(torch_python_obj SYSTEM PRIVATE
      ${PYTHON_INCLUDE_DIRS}
      ${pybind11_INCLUDE_DIRS})

  if(HAVE_SOVERSION)
    set_target_properties(torch_python_obj PROPERTIES
        VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
  endif()
  add_dependencies(torch_python_obj torch_python_stubs)

  # Required workaround for generated sources
  # See https://samthursfield.wordpress.com/2015/11/21/cmake-dependencies-between-targets-and-files-and-custom-commands/#custom-commands-in-different-directories
  add_dependencies(torch_python_obj generate-torch-sources)
  set_source_files_properties(
      ${GENERATED_THNN_SOURCES}
      ${GENERATED_CXX_PYTHON}
      PROPERTIES GENERATED TRUE
      )

  add_dependencies(torch_python_obj gen_torch_version)

endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947 in EmbeddingBag.cpp
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/utils/throughput_benchmark.cpp PROPERTIES COMPILE_FLAGS -Wno-attributes)
endif()

# coreml
if(USE_COREML_DELEGATE)
  list(APPEND TORCH_PYTHON_SRCS ${TORCH_SRC_DIR}/csrc/jit/backends/coreml/cpp/backend.cpp)
  list(APPEND TORCH_PYTHON_SRCS ${TORCH_SRC_DIR}/csrc/jit/backends/coreml/cpp/preprocess.cpp)
endif()


add_library(torch_python SHARED ${TORCH_PYTHON_SRCS})
# Avoid numpy for the DEPLOY build
if(USE_NUMPY)
  target_link_libraries(torch_python PRIVATE numpy::numpy)
  target_compile_definitions(torch_python PRIVATE USE_NUMPY)
endif()

if(HAVE_SOVERSION)
  set_target_properties(torch_python PROPERTIES
      VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
endif()
add_dependencies(torch_python torch_python_stubs)
add_dependencies(torch_python flatbuffers)


if(USE_PRECOMPILED_HEADERS)
  target_precompile_headers(torch_python PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:ATen/ATen.h>")
endif()

if(NOT MSVC)
  target_compile_options(torch_python PRIVATE -Wno-unused-variable)
endif()

# Required workaround for generated sources
# See https://samthursfield.wordpress.com/2015/11/21/cmake-dependencies-between-targets-and-files-and-custom-commands/#custom-commands-in-different-directories
add_dependencies(torch_python generate-torch-sources)
set_source_files_properties(
    ${GENERATED_THNN_SOURCES}
    ${GENERATED_CXX_PYTHON}
    PROPERTIES GENERATED TRUE
    )

# Disable certain warnings for GCC-9.X
if(CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0.0))
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/Module.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
  set_source_files_properties(
    ${TORCH_SRC_DIR}/csrc/init_flatbuffer_module.cpp
    PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/autograd/python_variable.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
endif()

# Preserve CUDA_GENCODE flags
if(USE_CUDA)
  torch_cuda_get_nvcc_gencode_flag(_ARCH_FLAGS)
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/cuda/Module.cpp PROPERTIES COMPILE_FLAGS "-DCUDA_ARCH_FLAGS=\"${_ARCH_FLAGS_readable}\"")
endif()

# Preserve HIP arch flags
if(USE_ROCM)
  string(REPLACE ";" " " PYTORCH_ROCM_ARCH_readable "${PYTORCH_ROCM_ARCH}")
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/cuda/Module.cpp PROPERTIES COMPILE_FLAGS "-DCUDA_ARCH_FLAGS=\"${PYTORCH_ROCM_ARCH_readable}\"")
endif()

target_compile_definitions(torch_python PRIVATE "-DTHP_BUILD_MAIN_LIB")

target_link_libraries(torch_python PRIVATE torch_library ${TORCH_PYTHON_LINK_LIBRARIES})

target_compile_definitions(torch_python PRIVATE ${TORCH_PYTHON_COMPILE_DEFINITIONS})

target_compile_definitions(torch_python PUBLIC ${TORCH_PYTHON_PUBLIC_COMPILE_DEFINITIONS})

target_compile_options(torch_python PRIVATE ${TORCH_PYTHON_COMPILE_OPTIONS})

target_include_directories(torch_python PUBLIC ${TORCH_PYTHON_INCLUDE_DIRECTORIES})

if(BUILD_ONEDNN_GRAPH)
  target_compile_definitions(torch_python PRIVATE "-DBUILD_ONEDNN_GRAPH")
  target_compile_definitions(torch_cpu PRIVATE "-DBUILD_ONEDNN_GRAPH")
endif()

if(NOT TORCH_PYTHON_LINK_FLAGS STREQUAL "")
    set_target_properties(torch_python PROPERTIES LINK_FLAGS ${TORCH_PYTHON_LINK_FLAGS})
endif()

install(TARGETS torch_python DESTINATION "${TORCH_INSTALL_LIB_DIR}")

# Generate torch/version.py from the appropriate CMake cache variables.
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(TORCH_VERSION_DEBUG 1)
else()
  set(TORCH_VERSION_DEBUG 0)
endif()

add_custom_command(
  OUTPUT ${TORCH_SRC_DIR}/version.py
  COMMAND
    "${PYTHON_EXECUTABLE}" -c \"from pathlib import Path\; Path('${TOOLS_PATH}/generate_torch_version.py').touch()\"
  COMMAND
    "${PYTHON_EXECUTABLE}" ${TOOLS_PATH}/generate_torch_version.py
      --is_debug=${TORCH_VERSION_DEBUG}
      --cuda_version=${CUDA_VERSION}
      --hip_version=${HIP_VERSION}
  DEPENDS ${TOOLS_PATH}/generate_torch_version.py
  WORKING_DIRECTORY ${TORCH_ROOT}
)
add_custom_target(
  gen_torch_version ALL
  DEPENDS ${TORCH_SRC_DIR}/version.py
)
add_dependencies(torch_python gen_torch_version)

# Skip building this library under MacOS, since it is currently failing to build on Mac
# Github issue #61930
if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # Add Android Nnapi delegate library
  add_library(nnapi_backend SHARED
          ${TORCH_SRC_DIR}/csrc/jit/backends/nnapi/nnapi_backend_lib.cpp
          ${TORCH_SRC_DIR}/csrc/jit/backends/nnapi/nnapi_backend_preprocess.cpp
          )
  # Pybind11 requires explicit linking of the torch_python library
  target_link_libraries(nnapi_backend PRIVATE torch torch_python pybind::pybind11)
endif()
