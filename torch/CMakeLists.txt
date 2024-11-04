# This file used to build libtorch.so.
# Now it only builds the Torch python bindings.

if(NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
  project(torch CXX C)
  find_package(torch REQUIRED)
  option(USE_CUDA "Use CUDA" ON)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(NOT BUILD_PYTHON)
  return()
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
    ${TORCH_ROOT}/third_party/cpp-httplib
    ${TORCH_ROOT}/third_party/nlohmann/include

    ${TORCH_SRC_DIR}/csrc
    ${TORCH_SRC_DIR}/csrc/api/include
    ${TORCH_SRC_DIR}/lib
    )

list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES ${LIBSHM_SRCDIR})

set(TORCH_PYTHON_LINK_LIBRARIES
    Python::Module
    pybind::pybind11
    opentelemetry::api
    httplib
    nlohmann
    shm
    fmt::fmt-header-only
    ATEN_CPU_FILES_GEN_LIB)

if(USE_ASAN AND TARGET Sanitizer::address)
  list(APPEND TORCH_PYTHON_LINK_LIBRARIES Sanitizer::address)
endif()
if(USE_ASAN AND TARGET Sanitizer::undefined)
  list(APPEND TORCH_PYTHON_LINK_LIBRARIES Sanitizer::undefined)
endif()
if(USE_TSAN AND TARGET Sanitizer::thread)
  list(APPEND TORCH_PYTHON_LINK_LIBRARIES Sanitizer::thread)
endif()

set(TORCH_PYTHON_COMPILE_DEFINITIONS)

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
      -Wno-strict-aliasing)
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
        list(APPEND TORCH_PYTHON_LINK_LIBRARIES torch::cudnn)
        list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUDNN)
    endif()
    if(USE_CUSPARSELT)
        list(APPEND TORCH_PYTHON_LINK_LIBRARIES torch::cusparselt)
        list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUSPARSELT)
    endif()

    if(TARGET torch::nvtx3)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES torch::nvtx3)
    else()
      if(TARGET torch::nvtoolsext)
        list(APPEND TORCH_PYTHON_LINK_LIBRARIES torch::nvtoolsext)
      endif()
    endif()
endif()

if(USE_ROCM)
    append_filelist("libtorch_python_cuda_core_sources" TORCH_PYTHON_SRCS)
    list(APPEND TORCH_PYTHON_SRCS ${GENERATED_THNN_CXX_CUDA})

    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS
      USE_ROCM
      __HIP_PLATFORM_AMD__
      )
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${ROCM_ROCTX_LIB})
endif()

if(USE_XPU)
    include(${TORCH_ROOT}/cmake/public/xpu.cmake)
    append_filelist("libtorch_python_xpu_sources" TORCH_PYTHON_SRCS)

    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_XPU)
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

if(USE_CUSPARSELT)
    list(APPEND TORCH_PYTHON_SRCS ${TORCH_SRC_DIR}/csrc/cuda/shared/cusparselt.cpp)
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
    "${Python_EXECUTABLE}" -mtools.pyi.gen_pyi
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
    "${Python_EXECUTABLE}" ${TORCH_SRC_DIR}/utils/data/datapipes/gen_pyi.py
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
    if(CMAKE_COMPILER_IS_GNUCXX)
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
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES MPI::MPI_CXX)
    endif()
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_C10D)

endif()

if(USE_NCCL AND NOT WIN32)
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/python_nccl.cpp)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_NCCL)
endif()

if(NOT MSVC)
  # cudaProfilerInitialize must go away
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/cuda/shared/cudart.cpp PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")
endif()

# coreml
if(USE_COREML_DELEGATE)
  list(APPEND TORCH_PYTHON_SRCS ${TORCH_SRC_DIR}/csrc/jit/backends/coreml/cpp/backend.cpp)
  list(APPEND TORCH_PYTHON_SRCS ${TORCH_SRC_DIR}/csrc/jit/backends/coreml/cpp/preprocess.cpp)
endif()


add_library(torch_python SHARED ${TORCH_PYTHON_SRCS})

if(CAFFE2_USE_MKL AND BUILD_LIBTORCHLESS)

  # Use the RPATH of the linked libraries
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  # we need to explicitly link caffe2::mkl in order to have the
  # correct RPATH in torch_python for the split build
  target_link_libraries(torch_python PRIVATE caffe2::mkl)
endif()

add_dependencies(torch_python onnx_proto)
# Avoid numpy for the DEPLOY build
if(USE_NUMPY)
  target_link_libraries(torch_python PRIVATE Python::NumPy)
  target_compile_definitions(torch_python PRIVATE USE_NUMPY)
endif()

if(USE_CUFILE AND NOT USE_ROCM)
  target_compile_definitions(torch_python PRIVATE USE_CUFILE)
endif()

if(HAVE_SOVERSION)
  set_target_properties(torch_python PROPERTIES
      VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
endif()

# in case of the split build we need to add compile definitions
if(BUILD_LIBTORCHLESS)

  if(USE_UCC)
    target_link_libraries(torch_python PRIVATE __caffe2_ucc)
    target_compile_definitions(torch_python PRIVATE USE_UCC)
  endif()

  if(USE_UCC AND USE_C10D_UCC)
    target_compile_definitions(torch_python PRIVATE USE_C10D_UCC)
  endif()

  if(USE_NCCL AND USE_C10D_NCCL)
    target_compile_definitions(torch_python PRIVATE USE_C10D_NCCL)
  endif()

  if(USE_DISTRIBUTED)
    target_compile_definitions(torch_python PRIVATE USE_DISTRIBUTED)
  endif()

  if(USE_MPI AND USE_C10D_MPI)
    target_compile_definitions(torch_python PRIVATE USE_C10D_MPI)
  endif()

  if(USE_GLOO AND USE_C10D_GLOO)
    target_compile_definitions(torch_python PRIVATE USE_C10D_GLOO)
  endif()

  if(NOT WIN32)
    target_compile_definitions(torch_python PRIVATE USE_RPC)
  endif()

  if(USE_TENSORPIPE)
    target_compile_definitions(torch_python PRIVATE USE_TENSORPIPE)
  endif()

  set(EXPERIMENTAL_SINGLE_THREAD_POOL "0" CACHE STRING
  "Experimental option to use a single thread pool for inter- and intra-op parallelism")
  if("${EXPERIMENTAL_SINGLE_THREAD_POOL}")
    target_compile_definitions(torch_python PRIVATE "-DAT_EXPERIMENTAL_SINGLE_THREAD_POOL=1")
  endif()

endif()

add_dependencies(torch_python torch_python_stubs)
add_dependencies(torch_python flatbuffers)


if(USE_PRECOMPILED_HEADERS)
  target_precompile_headers(torch_python PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:ATen/ATen.h>")
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
if(CMAKE_COMPILER_IS_GNUCXX)
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/Module.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
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

target_link_libraries(torch_python PRIVATE ${TORCH_LIB} ${TORCH_PYTHON_LINK_LIBRARIES})

target_compile_definitions(torch_python PRIVATE ${TORCH_PYTHON_COMPILE_DEFINITIONS})

target_compile_options(torch_python PRIVATE ${TORCH_PYTHON_COMPILE_OPTIONS})

target_include_directories(torch_python PUBLIC ${TORCH_PYTHON_INCLUDE_DIRECTORIES})

if(USE_UCC)
  target_link_libraries(torch_python PRIVATE __caffe2_ucc)
  target_compile_definitions(torch_python PRIVATE USE_UCC)
endif()

if(BUILD_ONEDNN_GRAPH)
  target_compile_definitions(torch_python PRIVATE "-DBUILD_ONEDNN_GRAPH")
  if(NOT BUILD_LIBTORCHLESS)
    target_compile_definitions(torch_cpu PRIVATE "-DBUILD_ONEDNN_GRAPH")
  endif()
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "RelWithAssert")
   # Workaround numerous decret-without-a-gil warnings from JIT
   # see https://github.com/pytorch/pytorch/issues/130073
   target_compile_definitions(torch_python PRIVATE "-DPYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF")
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
  COMMAND "${CMAKE_COMMAND}" -E touch "${TOOLS_PATH}/generate_torch_version.py"
  COMMAND
    "${Python_EXECUTABLE}" "${TOOLS_PATH}/generate_torch_version.py"
      --is-debug=${TORCH_VERSION_DEBUG}
      --cuda-version=${CUDA_VERSION}
      --hip-version=${HIP_VERSION}
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
  if(BUILD_LIBTORCHLESS)
    target_link_libraries(nnapi_backend PRIVATE ${TORCH_LIB} torch_python pybind::pybind11)
  else()
    target_link_libraries(nnapi_backend PRIVATE torch torch_python pybind::pybind11)
  endif()
endif()

set(TORCH_PYTHON_COMPILE_OPTIONS ${TORCH_PYTHON_COMPILE_OPTIONS} PARENT_SCOPE)
set(TORCH_PYTHON_LINK_FLAGS ${TORCH_PYTHON_LINK_FLAGS} PARENT_SCOPE)
