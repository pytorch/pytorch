
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

if(${CMAKE_VERSION} VERSION_LESS "2.8.12")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else(${CMAKE_VERSION} VERSION_LESS "2.8.12")
  if(${CMAKE_VERSION} VERSION_LESS "3.1")
    add_compile_options(-std=c++11) # CMake 2.8.12 to 3.1
  endif(${CMAKE_VERSION} VERSION_LESS "3.1")
endif(${CMAKE_VERSION} VERSION_LESS "2.8.12")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wno-vla")

################################################################################
# Helper functions
################################################################################

FUNCTION(EXCLUDE_DIR list_name dir_name)
  # A helper that excludes all files that contain dir_name in their file path
  SET(local_list ${${list_name}})
  FOREACH(source ${local_list})
    IF(${source} MATCHES ${dir_name})
      MESSAGE(STATUS "Excluding " ${source} " from the build")
      LIST(REMOVE_ITEM local_list ${source})
    ENDIF()
  ENDFOREACH()
  SET(${list_name} ${local_list} PARENT_SCOPE)
ENDFUNCTION()

function(filter_list output input)
    unset(result)
    foreach(filename ${${input}})
        foreach(pattern ${ARGN})
            if("${filename}" MATCHES "${pattern}")
                list(APPEND result "${filename}")
            endif()
        endforeach()
    endforeach()
    set(${output} ${result} PARENT_SCOPE)
endfunction()

IF(NOT Torch_FOUND)
  FIND_PACKAGE(Torch REQUIRED)
ENDIF()

IF(NOT TH_LIBRARIES)
  SET(TH_LIBRARIES "TH")
ENDIF(NOT TH_LIBRARIES)
MESSAGE(STATUS "TH_LIBRARIES: ${TH_LIBRARIES}")

IF(NOT THS_LIBRARIES)
  SET(THS_LIBRARIES "THS")
ENDIF()

IF(NOT THNN_LIBRARIES)
  SET(THNN_LIBRARIES "THNN")
ENDIF(NOT THNN_LIBRARIES)
MESSAGE(STATUS "THNN_LIBRARIES: ${THNN_LIBRARIES}")

IF(NOT CUDA_FOUND)
  MESSAGE(STATUS "ignoring CUDA")
  SET(CUDA_FLAG -n)
ELSE()
  ADD_DEFINITIONS(-DAT_CUDA_ENABLED)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

  IF(NOT THC_LIBRARIES)
    SET(THC_LIBRARIES "THC")
  ENDIF(NOT THC_LIBRARIES)
  MESSAGE(STATUS "THC_LIBRARIES: ${THC_LIBRARIES}")

  IF(NOT THCS_LIBRARIES)
    SET(THCS_LIBRARIES "THCS")
  ENDIF(NOT THCS_LIBRARIES)
  MESSAGE(STATUS "THCS_LIBRARIES: ${THCS_LIBRARIES}")

  IF(NOT THCUNN_LIBRARIES)
    SET(THCUNN_LIBRARIES "THCUNN")
  ENDIF(NOT THCUNN_LIBRARIES)
  MESSAGE(STATUS "THCUNN_LIBRARIES: ${THCUNN_LIBRARIES}")
ENDIF()

include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
"
#include <iostream>
#include <iomanip>

int main()
{
  std::cout << std::defaultfloat;
  return 0;
}
"
HAS_DEFAULTFLOAT
)

IF(HAS_DEFAULTFLOAT)
  ADD_DEFINITIONS(-DHAS_DEFAULTFLOAT)
ENDIF()

# Can be compiled standalone
IF(NOT TENSOR_LIB_INSTALL_BIN_DIR OR NOT TENSOR_LIB_INSTALL_LIB_DIR OR NOT TENSOR_LIB_INSTALL_INCLUDE_DIR)
  SET(TENSOR_LIB_INSTALL_BIN_DIR "bin" CACHE PATH "TENSOR_LIB install binary subdirectory")
  SET(TENSOR_LIB_INSTALL_LIB_DIR "lib" CACHE PATH "TENSOR_LIB install library subdirectory")
  SET(TENSOR_LIB_INSTALL_INCLUDE_DIR "include" CACHE PATH "TENSOR_LIB install include subdirectory")
ENDIF()

FILE(GLOB base_h RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")
FILE(GLOB base_cpp RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

FILE(GLOB all_python RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.py")

IF(NOT DEFINED cwrap_files)
  SET(CWRAP_FILES_BASE ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc  )
  SET(cwrap_files
  # ${CWRAP_FILES_BASE}/cudnn/cuDNN.cwrap
    ${CWRAP_FILES_BASE}/generic/TensorMethods.cwrap
  # ${CWRAP_FILES_BASE}/generic/methods/SparseTensor.cwrap
    ${CWRAP_FILES_BASE}/generic/methods/Tensor.cwrap
    ${CWRAP_FILES_BASE}/generic/methods/TensorApply.cwrap
    ${CWRAP_FILES_BASE}/generic/methods/TensorCompare.cwrap
    ${CWRAP_FILES_BASE}/generic/methods/TensorCuda.cwrap
    ${CWRAP_FILES_BASE}/generic/methods/TensorMath.cwrap
    ${CWRAP_FILES_BASE}/generic/methods/TensorRandom.cwrap
  #  ${CWRAP_FILES_BASE}/generic/methods/TensorSerialization.cwrap
    ${CMAKE_CURRENT_SOURCE_DIR}/Local.cwrap
    ${CMAKE_CURRENT_SOURCE_DIR}/../THNN/generic/THNN.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../THCUNN/generic/THCUNN.h
  )
ENDIF()

EXECUTE_PROCESS(
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/gen.py ${CUDA_FLAG} -s ${CMAKE_CURRENT_SOURCE_DIR} --print-dependencies ${cwrap_files}
    # user stderr rather than stdout so we can still debug the script with print
    ERROR_VARIABLE generated_cpp
    RESULT_VARIABLE RETURN_VALUE
)
if (NOT RETURN_VALUE EQUAL 0)
    message(STATUS ${generated_cpp})
    message(FATAL_ERROR "Failed to get generated_cpp list")
endif()

FILE(GLOB_RECURSE all_templates "templates/*")

FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ATen)

ADD_CUSTOM_COMMAND(OUTPUT ${generated_cpp}
COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/gen.py ${CUDA_FLAG} -s ${CMAKE_CURRENT_SOURCE_DIR} ${cwrap_files}
DEPENDS ${all_python} ${all_templates} ${cwrap_files})

SET(all_cpp ${base_cpp} ${generated_cpp})
filter_list(generated_h generated_cpp "\\.h$")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/..)
# so the build can find the generated header files
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
ADD_LIBRARY(ATen SHARED ${all_cpp})
SET_TARGET_PROPERTIES(ATen PROPERTIES VERSION 1 SOVERSION 1)

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.1")
    SET_PROPERTY(TARGET ATen PROPERTY CXX_STANDARD 11)
endif(NOT ${CMAKE_VERSION} VERSION_LESS "3.1")

TARGET_LINK_LIBRARIES(ATen ${TH_LIBRARIES} ${THNN_LIBRARIES} ${THS_LIBRARIES})
IF(CUDA_FOUND)
  TARGET_LINK_LIBRARIES(ATen ${THC_LIBRARIES} ${THCUNN_LIBRARIES} ${THCS_LIBRARIES})
  TARGET_LINK_LIBRARIES(ATen ${CUDA_LIBRARIES})
ENDIF()

INSTALL(TARGETS ATen
  RUNTIME DESTINATION "${TENSOR_LIB_INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${TENSOR_LIB_INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${TENSOR_LIB_INSTALL_LIB_DIR}")

ADD_EXECUTABLE(scalar_test test/scalar_test.cpp)
TARGET_LINK_LIBRARIES(scalar_test ATen)

ADD_EXECUTABLE(basic test/basic.cpp)
TARGET_LINK_LIBRARIES(basic ATen)

add_executable(atest test/atest.cpp)
target_link_libraries(atest ATen)

FOREACH(HEADER ${base_h})
  INSTALL(FILES ${HEADER} DESTINATION ${TENSOR_LIB_INSTALL_INCLUDE_DIR}/ATen)
ENDFOREACH()
FOREACH(HEADER ${generated_h})
  INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/${HEADER}
  DESTINATION ${TENSOR_LIB_INSTALL_INCLUDE_DIR}/ATen)
ENDFOREACH()
