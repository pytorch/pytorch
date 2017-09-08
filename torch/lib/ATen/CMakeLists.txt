
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

IF ($ENV{TH_BINARY_BUILD})
  MESSAGE(STATUS "TH_BINARY_BUILD detected. Statically linking libstdc++")
  SET(CMAKE_CXX_FLAGS "-static-libstdc++ ${CMAKE_CXX_FLAGS}")
  IF (UNIX AND NOT APPLE)
    # hiding statically linked library symbols, this flag is not available for the linker under MACOSX
    SET(CMAKE_CXX_FLAGS "-Wl,--exclude-libs,libstdc++.a ${CMAKE_CXX_FLAGS}")
  ENDIF(UNIX AND NOT APPLE)
ENDIF()

IF(NO_CUDA)
  MESSAGE(STATUS "ignoring CUDA")
  SET(CUDA_FLAG -n)
ELSE()
  FIND_PACKAGE(CUDA 5.5)
  IF(CUDA_FOUND)
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
ENDIF()

# Can be compiled standalone
IF(NOT AT_INSTALL_BIN_DIR OR NOT AT_INSTALL_LIB_DIR OR NOT AT_INSTALL_INCLUDE_DIR OR NOT AT_INSTALL_SHARE_DIR)
  SET(AT_INSTALL_BIN_DIR "bin" CACHE PATH "AT install binary subdirectory")
  SET(AT_INSTALL_LIB_DIR "lib" CACHE PATH "AT install library subdirectory")
  SET(AT_INSTALL_INCLUDE_DIR "include" CACHE PATH "AT install include subdirectory")
  SET(AT_INSTALL_SHARE_DIR "share" CACHE PATH "AT install include subdirectory")
ENDIF()

FILE(GLOB base_h RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")
FILE(GLOB base_cpp RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

FILE(GLOB all_python RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.py")

IF(NOT DEFINED cwrap_files)
  IF(NOT DEFINED CWRAP_FILES_BASE)
    SET(CWRAP_FILES_BASE ${CMAKE_CURRENT_SOURCE_DIR}/../..  )
  ENDIF()
  SET(cwrap_files
  # ${CWRAP_FILES_BASE}/csrc/cudnn/cuDNN.cwrap
    ${CWRAP_FILES_BASE}/csrc/generic/TensorMethods.cwrap
  # ${CWRAP_FILES_BASE}/csrc/generic/methods/SparseTensor.cwrap
    ${CWRAP_FILES_BASE}/csrc/generic/methods/Tensor.cwrap
    ${CWRAP_FILES_BASE}/csrc/generic/methods/TensorApply.cwrap
    ${CWRAP_FILES_BASE}/csrc/generic/methods/TensorCompare.cwrap
    ${CWRAP_FILES_BASE}/csrc/generic/methods/TensorCuda.cwrap
    ${CWRAP_FILES_BASE}/csrc/generic/methods/TensorMath.cwrap
    ${CWRAP_FILES_BASE}/csrc/generic/methods/TensorRandom.cwrap
  #  ${CWRAP_FILES_BASE}/csrc/generic/methods/TensorSerialization.cwrap
    ${CMAKE_CURRENT_SOURCE_DIR}/Local.cwrap
    ${CWRAP_FILES_BASE}/lib/THNN/generic/THNN.h
    ${CWRAP_FILES_BASE}/lib/THCUNN/generic/THCUNN.h
  )
ENDIF()

SET(GEN_COMMAND
    python ${CMAKE_CURRENT_SOURCE_DIR}/gen.py ${CUDA_FLAG}
    -s ${CMAKE_CURRENT_SOURCE_DIR}
    ${cwrap_files}
)

EXECUTE_PROCESS(
    COMMAND ${GEN_COMMAND} --output-dependencies ${CMAKE_CURRENT_BINARY_DIR}/generated_cpp.txt
    RESULT_VARIABLE RETURN_VALUE
)
if (NOT RETURN_VALUE EQUAL 0)
    message(STATUS ${generated_cpp})
    message(FATAL_ERROR "Failed to get generated_cpp list")
endif()
file(READ ${CMAKE_CURRENT_BINARY_DIR}/generated_cpp.txt generated_cpp)

FILE(GLOB_RECURSE all_templates "templates/*")

FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ATen)

ADD_CUSTOM_COMMAND(OUTPUT ${generated_cpp}
COMMAND ${GEN_COMMAND}
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
  RUNTIME DESTINATION "${AT_INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${AT_INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${AT_INSTALL_LIB_DIR}")

FOREACH(HEADER ${base_h})
  INSTALL(FILES ${HEADER} DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen)
ENDFOREACH()
FOREACH(HEADER ${generated_h})
  INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/${HEADER}
  DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen)
ENDFOREACH()
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/ATen/Declarations.yaml
  DESTINATION ${AT_INSTALL_SHARE_DIR}/ATen)
