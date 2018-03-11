CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)
CMAKE_POLICY(VERSION 3.0)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

FIND_PATH(ATEN_INCLUDE_DIR ATen/ATen.h PATHS "${CMAKE_INSTALL_PREFIX}/include/" NO_DEFAULT_PATH)
FIND_LIBRARY(ATEN_LIBRARY ATen PATHS "${CMAKE_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)

FIND_PATH(TORCH_SRC_DIR torch.h PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../torch" NO_DEFAULT_PATH)

IF(NOT NO_CUDA)
  set(CMAKE_MODULE_PATH
    ${TORCH_SRC_DIR}/../aten/cmake
    ${TORCH_SRC_DIR}/../aten/cmake/FindCUDA
    /usr/lib/x86_64-linux-gnu/
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TH/cmake
    ${CMAKE_MODULE_PATH})
  SET(CMAKE_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/ ${CMAKE_LIBRARY_PATH})

  IF(NOT CUDA_FOUND)
    FIND_PACKAGE(CUDA 5.5)
  ENDIF()

  FIND_PACKAGE(MAGMA)
  IF(CUDA_FOUND AND MAGMA_FOUND)
    INCLUDE_DIRECTORIES("${MAGMA_INCLUDE_DIR}")
    SET(CMAKE_REQUIRED_INCLUDES "${MAGMA_INCLUDE_DIR};${CUDA_INCLUDE_DIRS}")
    INCLUDE(CheckPrototypeDefinition)
    check_prototype_definition(magma_get_sgeqrf_nb
     "magma_int_t magma_get_sgeqrf_nb( magma_int_t m, magma_int_t n );"
     "0"
     "magma.h"
      MAGMA_V2)
    IF (MAGMA_V2)
      add_definitions(-DMAGMA_V2)
    ENDIF (MAGMA_V2)

    SET(USE_MAGMA 1)
    MESSAGE(STATUS "Compiling with MAGMA support")
    MESSAGE(STATUS "MAGMA INCLUDE DIRECTORIES: ${MAGMA_INCLUDE_DIR}")
    MESSAGE(STATUS "MAGMA LIBRARIES: ${MAGMA_LIBRARIES}")
    MESSAGE(STATUS "MAGMA V2 check: ${MAGMA_V2}")
  ELSE()
    MESSAGE(STATUS "MAGMA not found. Compiling without MAGMA support")
  ENDIF()
ENDIF()

ADD_DEFINITIONS(-DNO_PYTHON)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")

IF(NOT TORCH_INSTALL_BIN_DIR)
  SET(TORCH_INSTALL_BIN_DIR bin)
ENDIF(NOT TORCH_INSTALL_BIN_DIR)

IF(NOT TORCH_INSTALL_INCLUDE_DIR)
  SET(TORCH_INSTALL_INCLUDE_DIR include/libtorch)
ENDIF(NOT TORCH_INSTALL_INCLUDE_DIR)

IF(NOT TORCH_INSTALL_LIB_DIR)
  SET(TORCH_INSTALL_LIB_DIR lib)
ENDIF(NOT TORCH_INSTALL_LIB_DIR)

IF(NOT NO_CUDA)
  ADD_DEFINITIONS( -DWITH_CUDA )

  SET(TORCH_CUDA_LIBRARIES
    cuda
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvrtc.so
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so
    ${CUDA_LIBRARIES}
  )

ENDIF(NOT NO_CUDA)

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

SET(TORCH_SRCS
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
  )

ADD_LIBRARY(torch SHARED ${TORCH_SRCS})

TARGET_LINK_LIBRARIES(torch
  ${TORCH_CUDA_LIBRARIES}
  ${ATEN_LIBRARY}
)

TARGET_INCLUDE_DIRECTORIES(torch
  PUBLIC
  "${ATEN_INCLUDE_DIR}"
  "${ATEN_INCLUDE_DIR}/TH"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${TORCH_SRC_DIR}/../"
  "${TORCH_SRC_DIR}/lib/nanopb"
  "${CUDA_INCLUDE_DIRS}"
  )

SET_TARGET_PROPERTIES(torch PROPERTIES VERSION 1 SOVERSION 1)

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.1")
  SET_PROPERTY(TARGET torch PROPERTY CXX_STANDARD 11)
endif(NOT ${CMAKE_VERSION} VERSION_LESS "3.1")

install(DIRECTORY "${TORCH_SRC_DIR}/csrc"
        DESTINATION ${TORCH_INSTALL_INCLUDE_DIR}/torch
        FILES_MATCHING PATTERN "*.h")

INSTALL(TARGETS torch
  RUNTIME DESTINATION "${TORCH_INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${TORCH_INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${TORCH_INSTALL_LIB_DIR}")


SET(TORCH_TEST_SRCS
  ${TORCH_SRC_DIR}/csrc/jit/test_jit.cpp
  )

ADD_EXECUTABLE(test_jit ${TORCH_TEST_SRCS})

TARGET_LINK_LIBRARIES(test_jit torch)

TARGET_INCLUDE_DIRECTORIES(test_jit
  PUBLIC
  "${ATEN_INCLUDE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${TORCH_SRC_DIR}/../"
  "${TORCH_SRC_DIR}/../aten/src/ATen/utils/catch/single_include"
  "${CUDA_INCLUDE_DIRS}"
  )

INSTALL(TARGETS test_jit
  RUNTIME DESTINATION "${TORCH_INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${TORCH_INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${TORCH_INSTALL_LIB_DIR}")

