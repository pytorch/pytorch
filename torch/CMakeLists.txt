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
if(BUILD_TEST AND NOT USE_ROCM)
    add_definitions(-DBUILDING_TESTS)
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_ROOT}/test/cpp/jit/torch_python_test.cpp
      ${TENSOREXPR_TEST_SRCS_WITH_PADDED}
      ${JIT_TEST_SRCS}
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

    ${TORCH_ROOT}/third_party/gloo
    ${TORCH_ROOT}/third_party/onnx
    ${TORCH_ROOT}/third_party/tensorpipe
    ${pybind11_INCLUDE_DIRS}

    ${TORCH_SRC_DIR}/csrc
    ${TORCH_SRC_DIR}/csrc/api/include
    ${TORCH_SRC_DIR}/lib
    )


list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES ${LIBSHM_SRCDIR})

set(TORCH_PYTHON_LINK_LIBRARIES
    torch_library
    shm
    fmt::fmt-header-only)

set(TORCH_PYTHON_COMPILE_DEFINITIONS)

set(TORCH_PYTHON_COMPILE_OPTIONS)

set(TORCH_PYTHON_LINK_FLAGS "")

if(MSVC)
    string(APPEND TORCH_PYTHON_LINK_FLAGS " /NODEFAULTLIB:LIBCMT.LIB")
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${PYTHON_LIBRARIES})
    if(BUILD_TEST)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES onnx_library)
    endif(BUILD_TEST)
    if(NOT ${CMAKE_BUILD_TYPE} MATCHES "Release")
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
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/Module.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Storage.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Stream.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Event.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/utils.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/python_comm.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/serialization.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/shared/cudart.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/shared/nvtx.cpp
      ${GENERATED_THNN_CXX_CUDA}
      )
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUDA)

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

if(USE_CUDNN)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUDNN)

    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/shared/cudnn.cpp
      )
endif()

if(USE_NUMPY)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_NUMPY)
endif()

if(USE_ROCM)
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/Module.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Storage.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Stream.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Event.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/utils.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/python_comm.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/serialization.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/shared/cudart.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/shared/cudnn.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/shared/nvtx.cpp
      ${GENERATED_THNN_CXX_CUDA}
      )

    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS
      USE_ROCM
      __HIP_PLATFORM_HCC__
      )
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${ROCM_ROCTX_LIB})
    list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES ${roctracer_INCLUDE_DIRS})
endif()

if(USE_DISTRIBUTED)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_DISTRIBUTED)
    if(NOT MSVC)
      append_filelist("libtorch_python_distributed_sources" TORCH_PYTHON_SRCS)
      # Disable certain warnings for GCC-9.X
      if(CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0.0))
        set_source_files_properties(${TORCH_SRC_DIR}/csrc/distributed/autograd/init.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
        set_source_files_properties(${TORCH_SRC_DIR}/csrc/distributed/c10d/init.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
        set_source_files_properties(${TORCH_SRC_DIR}/csrc/distributed/rpc/init.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
        set_source_files_properties(${TORCH_SRC_DIR}/csrc/distributed/rpc/testing/init.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
      endif()
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES c10d)
      list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_C10D)
    endif()
endif()

if(USE_NCCL)
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/python_nccl.cpp)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_NCCL)
    list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES ${NCCL_INCLUDE_DIRS})
endif()

# In the most recent CMake versions, a new 'TRANSFORM' subcommand of 'list' allows much of the boilerplate of defining the lists
# of type stub files to be omitted.
# For comptability with older CMake versions, we omit it for now, but leave it as a comment in case comptability with the older
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
      --declarations-path "${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml"
    DEPENDS
    "${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml"
    "${TORCH_SRC_DIR}/_C/__init__.pyi.in"
    "${TORCH_SRC_DIR}/_C/_VariableFunctions.pyi.in"
    "${TORCH_SRC_DIR}/nn/functional.pyi.in"
    "${TOOLS_PATH}/pyi/gen_pyi.py"
    WORKING_DIRECTORY
    "${TORCH_ROOT}"
)

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

target_compile_definitions(torch_python PRIVATE "-DTHP_BUILD_MAIN_LIB")

target_link_libraries(torch_python ${TORCH_PYTHON_LINK_LIBRARIES})

target_compile_definitions(torch_python PRIVATE ${TORCH_PYTHON_COMPILE_DEFINITIONS})

target_compile_options(torch_python PRIVATE ${TORCH_PYTHON_COMPILE_OPTIONS})

target_include_directories(torch_python PUBLIC ${TORCH_PYTHON_INCLUDE_DIRECTORIES})


if(NOT TORCH_PYTHON_LINK_FLAGS STREQUAL "")
    set_target_properties(torch_python PROPERTIES LINK_FLAGS ${TORCH_PYTHON_LINK_FLAGS})
endif()

install(TARGETS torch_python DESTINATION "${TORCH_INSTALL_LIB_DIR}")
