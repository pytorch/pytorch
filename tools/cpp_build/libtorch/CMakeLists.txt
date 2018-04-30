cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
cmake_policy(VERSION 3.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (VERBOSE)
  message(STATUS "ATEN_PATH is ${ATEN_PATH}")
  message(STATUS "ATEN_BUILD_PATH is ${ATEN_BUILD_PATH}")
endif()

find_path(ATEN_INCLUDE_DIR ATen/ATen.h PATHS "${ATEN_PATH}/src/" NO_DEFAULT_PATH)
find_path(ATEN_BUILD_INCLUDE_DIR ATen/Type.h PATHS "${ATEN_BUILD_PATH}/src/ATen" NO_DEFAULT_PATH)
find_library(ATEN_LIBRARY ATen PATHS "${ATEN_BUILD_PATH}/src/ATen" NO_DEFAULT_PATH)

find_path(TORCH_SRC_DIR torch.h PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../torch" NO_DEFAULT_PATH)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT NO_CUDA)
  set(CMAKE_MODULE_PATH
    ${TORCH_SRC_DIR}/../aten/cmake
    ${TORCH_SRC_DIR}/../aten/cmake/FindCUDA
    /usr/lib/x86_64-linux-gnu/
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TH/cmake
    ${CMAKE_MODULE_PATH})
  set(CMAKE_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/ ${CMAKE_LIBRARY_PATH})

  if(NOT CUDA_FOUND)
    find_package(CUDA 5.5)
  endif()

  find_package(MAGMA)
  if(CUDA_FOUND AND MAGMA_FOUND)
    include_directories("${MAGMA_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_INCLUDES "${MAGMA_INCLUDE_DIR};${CUDA_INCLUDE_DIRS}")
    include(CheckPrototypeDefinition)
    check_prototype_definition(magma_get_sgeqrf_nb
     "magma_int_t magma_get_sgeqrf_nb( magma_int_t m, magma_int_t n );"
     "0"
     "magma.h"
      MAGMA_V2)
    IF (MAGMA_V2)
      add_definitions(-DMAGMA_V2)
    endif (MAGMA_V2)

    set(USE_MAGMA 1)
    if(VERBOSE)
      message(STATUS "Compiling with MAGMA support")
      message(STATUS "MAGMA INCLUDE DIRECTORIES: ${MAGMA_INCLUDE_DIR}")
      message(STATUS "MAGMA LIBRARIES: ${MAGMA_LIBRARIES}")
      message(STATUS "MAGMA V2 check: ${MAGMA_V2}")
    endif()
  else()
    message(STATUS "MAGMA not found. Compiling without MAGMA support")
  endif()
endif()

add_definitions(-DNO_PYTHON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")

if(NOT TORCH_INSTALL_BIN_DIR)
  set(TORCH_INSTALL_BIN_DIR bin)
endif()

if(NOT TORCH_INSTALL_INCLUDE_DIR)
  set(TORCH_INSTALL_INCLUDE_DIR include/libtorch)
endif()

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()

if(NOT NO_CUDA)
  add_definitions( -DWITH_CUDA )

  set(TORCH_CUDA_LIBRARIES
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libcuda.so
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvrtc.so
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so
    ${CUDA_LIBRARIES}
  )

  set(CUDA_INCLUDE_DIRS
      ${CUDA_INCLUDE_DIRS}
      "${ATEN_BUILD_PATH}/src/THC"
      "${ATEN_PATH}/src/THC")
endif()

# RPATH stuff
# see https://cmake.org/Wiki/CMake_RPATH_handling
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-missing-field-initializers -Wno-type-limits -Wno-unused-parameter -Wno-vla ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "-fexceptions ${CMAKE_C_FLAGS}")
if ($ENV{WERROR})
  set(CMAKE_CXX_FLAGS "-Werror ${CMAKE_CXX_FLAGS}")
endif()

set(TORCH_SRCS
  ${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/generated/Functions.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/profiler.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/saved_variable.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/grad_mode.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/function.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/input_buffer.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/utils.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/special.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/basic_ops.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/accumulate_grad.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/functions/tensor.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/variable.cpp
  ${TORCH_SRC_DIR}/csrc/autograd/engine.cpp
  ${TORCH_SRC_DIR}/csrc/assertions.cpp
  ${TORCH_SRC_DIR}/csrc/utils/variadic.cpp
  ${TORCH_SRC_DIR}/csrc/jit/generated/aten_dispatch.cpp
  ${TORCH_SRC_DIR}/csrc/jit/variable_flags.cpp
  ${TORCH_SRC_DIR}/csrc/jit/interpreter.cpp
  ${TORCH_SRC_DIR}/csrc/jit/ir.cpp
  ${TORCH_SRC_DIR}/csrc/jit/graph_executor.cpp
  ${TORCH_SRC_DIR}/csrc/jit/fusion_compiler.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/graph_fuser.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/common_subexpression_elimination.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/shape_analysis.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/canonicalize.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/dead_code_elimination.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/lower_tuples.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/peephole.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/inplace_check.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/batch_mm.cpp
  ${TORCH_SRC_DIR}/csrc/jit/passes/create_autodiff_subgraphs.cpp
  ${TORCH_SRC_DIR}/csrc/jit/interned_strings.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/compiler.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/lexer.cpp
  ${TORCH_SRC_DIR}/csrc/jit/script/module.cpp
  ${TORCH_SRC_DIR}/csrc/jit/tracer.cpp
  ${TORCH_SRC_DIR}/csrc/jit/tracer_state.cpp
  ${TORCH_SRC_DIR}/csrc/jit/autodiff.cpp
  ${TORCH_SRC_DIR}/csrc/jit/type.cpp
  ${TORCH_SRC_DIR}/csrc/jit/interpreter_autograd_function.cpp
  ${TORCH_SRC_DIR}/csrc/Exceptions.cpp
  ${TORCH_SRC_DIR}/csrc/api/src/detail.cpp
  ${TORCH_SRC_DIR}/csrc/api/src/containers.cpp
  ${TORCH_SRC_DIR}/csrc/api/src/optimizers.cpp
  )

add_library(torch SHARED ${TORCH_SRCS})

target_compile_options(torch PRIVATE -Wall -Wextra)

if ($ENV{WERROR})
  target_compile_options(torch PRIVATE -Werror)
endif()

target_link_libraries(torch
  ${TORCH_CUDA_LIBRARIES}
  ${ATEN_LIBRARY}
)

set(COMMON_INCLUDES
  "${ATEN_INCLUDE_DIR}"
  "${ATEN_INCLUDE_DIR}/TH"
  "${ATEN_BUILD_INCLUDE_DIR}"
  "${ATEN_BUILD_PATH}/src/TH"
  "${TORCH_SRC_DIR}/csrc/api/"
  "${TORCH_SRC_DIR}/csrc/api/include"
  "${TORCH_SRC_DIR}/../third_party/cereal/include" # For cereal/
  "${TORCH_SRC_DIR}/../"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CUDA_INCLUDE_DIRS}")

target_include_directories(torch
  PUBLIC
  "${COMMON_INCLUDES}"
  "${TORCH_SRC_DIR}/../"
  "${TORCH_SRC_DIR}/../third_party/nanopb")

set_target_properties(torch PROPERTIES VERSION 1 SOVERSION 1)

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.1")
  set_property(TARGET torch PROPERTY CXX_STANDARD 11)
endif()

install(DIRECTORY "${TORCH_SRC_DIR}/csrc"
        DESTINATION ${TORCH_INSTALL_INCLUDE_DIR}/torch
        FILES_MATCHING PATTERN "*.h")

install(TARGETS torch
  RUNTIME DESTINATION "${TORCH_INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${TORCH_INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${TORCH_INSTALL_LIB_DIR}")

# JIT Tests. TODO: Put into test/cpp/jit folder

add_executable(test_jit ${TORCH_SRC_DIR}/csrc/jit/test_jit.cpp)

target_link_libraries(test_jit torch)

target_include_directories(test_jit PUBLIC
  "${TORCH_SRC_DIR}/../"
  "${TORCH_SRC_DIR}/../third_party/catch/single_include"
  "${COMMON_INCLUDES}")

# API Tests

set(TORCH_API_TEST_DIR "${TORCH_SRC_DIR}/../test/cpp/api")

add_executable(test_api
  ${TORCH_API_TEST_DIR}/main.cpp
  ${TORCH_API_TEST_DIR}/container.cpp
  ${TORCH_API_TEST_DIR}/misc.cpp
  ${TORCH_API_TEST_DIR}/rnn.cpp
  ${TORCH_API_TEST_DIR}/integration.cpp
  ${TORCH_API_TEST_DIR}/optim.cpp
  ${TORCH_API_TEST_DIR}/serialization.cpp
)

target_include_directories(test_api
  PUBLIC
  "${TORCH_SRC_DIR}/../third_party/catch/single_include")

target_link_libraries(test_api torch)
