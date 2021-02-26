# This file used to build libtorch.so.
# Now it only builds the Torch python bindings.

if(NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
  project(torch CXX C)
  find_package(torch REQUIRED)
  option(USE_CUDA "Use CUDA" ON)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(NOT BUILD_PYTHON)
  return()
endif()

if(USE_TBB)
include_directories(${TBB_ROOT_DIR}/include)
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
    ${pybind11_INCLUDE_DIRS}

    ${TORCH_SRC_DIR}/csrc
    ${TORCH_SRC_DIR}/csrc/api/include
    ${TORCH_SRC_DIR}/lib
    )


list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES ${LIBSHM_SRCDIR})

set(TORCH_PYTHON_LINK_LIBRARIES
    shm
    fmt::fmt-header-only)

set(TORCH_PYTHON_COMPILE_DEFINITIONS)

set(TORCH_PYTHON_PUBLIC_COMPILE_DEFINITIONS)

set(TORCH_PYTHON_COMPILE_OPTIONS)

set(TORCH_PYTHON_LINK_FLAGS "")

if(MSVC)
    string(APPEND TORCH_PYTHON_LINK_FLAGS " /NODEFAULTLIB:LIBCMT.LIB")
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${PYTHON_LIBRARIES})
    if(BUILD_TEST)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES onnx_library)
    endif(BUILD_TEST)
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

if(USE_CUDA)
    append_filelist("libtorch_python_cuda_core_sources" TORCH_PYTHON_SRCS)
    list(APPEND TORCH_PYTHON_SRCS ${GENERATED_THNN_CXX_CUDA})

    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUDA)
    if(USE_CUDNN)
        list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUDNN)
    endif()

    if(MSVC)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${NVTOOLEXT_HOME}/lib/x64/nvToolsExt64_1.lib)
      list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES "${NVTOOLEXT_HOME}/include")
    elseif(APPLE)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib/libnvToolsExt.dylib)
    else()
      find_library(LIBNVTOOLSEXT libnvToolsExt.so PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${LIBNVTOOLSEXT})
    endif()

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

if(USE_MLCOMPUTE)
    list(APPEND TORCH_PYTHON_SRCS ${MLC_PYTHON_SRCS})
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
)
# For Declarations.yaml dependency
add_dependencies(torch_python_stubs ATEN_CPU_FILES_GEN_TARGET)
add_custom_command(
    OUTPUT
    "${TORCH_SRC_DIR}/_C/__init__.pyi"
    "${TORCH_SRC_DIR}/_C/_VariableFunctions.pyi"
    "${TORCH_SRC_DIR}/nn/functional.pyi"
    COMMAND
    "${PYTHON_EXECUTABLE}" -mtools.pyi.gen_pyi
      --native-functions-path "aten/src/ATen/native/native_functions.yaml"
      --deprecated-functions-path "tools/autograd/deprecated.yaml"
    DEPENDS
    "${TORCH_SRC_DIR}/_C/__init__.pyi.in"
    "${TORCH_SRC_DIR}/_C/_VariableFunctions.pyi.in"
    "${TORCH_SRC_DIR}/nn/functional.pyi.in"
    "${TOOLS_PATH}/pyi/gen_pyi.py"
    WORKING_DIRECTORY
    "${TORCH_ROOT}"
)

# WARNING- any TORCH_PYTHON_COMPILE_DEFINITIONS above this line
#          affect both torch_python and DEPLOY interpreter.
if(USE_DEPLOY)
  add_library(torch_python_obj OBJECT ${TORCH_PYTHON_SRCS})
  target_compile_definitions(torch_python_obj PRIVATE "-DTHP_BUILD_MAIN_LIB -DUSE_DEPLOY")

  target_compile_definitions(torch_python_obj PRIVATE ${TORCH_PYTHON_COMPILE_DEFINITIONS})

  target_compile_definitions(torch_python_obj PUBLIC ${TORCH_PYTHON_PUBLIC_COMPILE_DEFINITIONS})

  target_compile_options(torch_python_obj PRIVATE ${TORCH_PYTHON_COMPILE_OPTIONS})

  target_include_directories(torch_python_obj PUBLIC ${TORCH_PYTHON_INCLUDE_DIRECTORIES})
  target_include_directories(torch_python_obj PRIVATE ../third_party/fmt/include)

  # need to specify the dependency so the generated headers exist,
  # missing dependency since torch_python_obj doesn't link onnx, the interpreter lib does.
  add_dependencies(torch_python_obj onnx)

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


# Avoid numpy for the DEPLOY build
if(USE_NUMPY)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_NUMPY)
endif()

if(USE_DISTRIBUTED)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_DISTRIBUTED)
    if(WIN32)
      append_filelist("libtorch_python_distributed_core_sources" TORCH_PYTHON_SRCS)
    else()
      list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_RPC)
      append_filelist("libtorch_python_distributed_sources" TORCH_PYTHON_SRCS)
    endif()
    # Disable certain warnings for GCC-9.X
    if(CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0.0))
      set_source_files_properties(${TORCH_SRC_DIR}/csrc/distributed/autograd/init.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
      set_source_files_properties(${TORCH_SRC_DIR}/csrc/distributed/rpc/init.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
      set_source_files_properties(${TORCH_SRC_DIR}/csrc/distributed/rpc/testing/init.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
      set_source_files_properties(${TORCH_SRC_DIR}/csrc/distributed/c10d/init.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
    endif()
    if(USE_TENSORPIPE)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES tensorpipe)
      list(APPEND TORCH_PYTHON_PUBLIC_COMPILE_DEFINITIONS USE_TENSORPIPE)
    endif()
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES c10d)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_C10D)
endif()

if(USE_NCCL AND NOT WIN32)
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/python_nccl.cpp)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_NCCL)
endif()

add_library(torch_python SHARED ${TORCH_PYTHON_SRCS})
if(HAVE_SOVERSION)
  set_target_properties(torch_python PROPERTIES
      VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
endif()
add_dependencies(torch_python torch_python_stubs)

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
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/autograd/python_variable.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
endif()

# Preserve CUDA_GENCODE flags
if(USE_CUDA)
  torch_cuda_get_nvcc_gencode_flag(_ARCH_FLAGS)
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/cuda/Module.cpp PROPERTIES COMPILE_FLAGS "-DCUDA_ARCH_FLAGS=\"${_ARCH_FLAGS_readable}\"")
endif()

target_compile_definitions(torch_python PRIVATE "-DTHP_BUILD_MAIN_LIB")

target_link_libraries(torch_python torch_library ${TORCH_PYTHON_LINK_LIBRARIES})

target_compile_definitions(torch_python PRIVATE ${TORCH_PYTHON_COMPILE_DEFINITIONS})

target_compile_definitions(torch_python PUBLIC ${TORCH_PYTHON_PUBLIC_COMPILE_DEFINITIONS})

target_compile_options(torch_python PRIVATE ${TORCH_PYTHON_COMPILE_OPTIONS})

target_include_directories(torch_python PUBLIC ${TORCH_PYTHON_INCLUDE_DIRECTORIES})


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
