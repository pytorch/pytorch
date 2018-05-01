CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

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

################################################################################
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

INCLUDE(CheckCXXSourceCompiles)

CHECK_CXX_SOURCE_COMPILES("
#include <thread>

thread_local int foo=1;

int main() {
  return 0;
}" HAS_THREAD_LOCAL)

IF(NOT HAS_THREAD_LOCAL)
    MESSAGE(FATAL_ERROR "thread_local not supported. THD requires a compiler"
                        " that supports thread_local. Please upgrade your "
                        "compiler. If you are on macOS, upgrade to "
                        "XCode 8 or newer.")
ENDIF(NOT HAS_THREAD_LOCAL)

FIND_PACKAGE(NCCL)

FIND_PACKAGE(MPI)

FIND_PACKAGE(Gloo)

FIND_PACKAGE(ATen REQUIRED)
INCLUDE_DIRECTORIES(${ATEN_INCLUDE_DIR})

IF(NO_CUDA)
  MESSAGE(STATUS "ignoring CUDA")
ELSE()
  FIND_PACKAGE(CUDA 7.5)
  IF(CUDA_FOUND)
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    LINK_DIRECTORIES("${CUDA_TOOLKIT_ROOT_DIR}/lib" "${CUDA_TOOLKIT_ROOT_DIR}/lib64")

    ADD_DEFINITIONS(-DWITH_CUDA=1)
  ENDIF()
ENDIF()

IF(MPI_FOUND)
  ADD_DEFINITIONS(-DWITH_MPI=1)
  MESSAGE(STATUS "MPI_LIBRARIES: ${MPI_LIBRARIES}")
ENDIF()

IF(GLOO_FOUND)
  ADD_DEFINITIONS(-DWITH_GLOO=1)
  MESSAGE(STATUS "Found Gloo, will compile with Gloo distributed backend")
  IF(WITH_GLOO_IBVERBS)
    MESSAGE(STATUS "Building the gloo backend with both TCP and infiniband support")
    ADD_DEFINITIONS(-DWITH_GLOO_IBVERBS=1)
  ELSE()
    MESSAGE(STATUS "Building the gloo backend with TCP support only")
  ENDIF()
ENDIF()

IF(NCCL_FOUND)
  MESSAGE(STATUS "NCCL_LIBRARIES: ${NCCL_LIBRARIES}")
  IF(NCCL_MAJOR_VERSION AND NOT (NCCL_MAJOR_VERSION LESS 2))
    MESSAGE(STATUS "NCCL Version 2 or higher found, will "
                   "compile with NCCL distributed backend")
    SET(DISTRIBUTED_NCCL_FOUND TRUE)
    ADD_DEFINITIONS(-DWITH_DISTRIBUTED_NCCL=1)
  ELSE()
    MESSAGE(STATUS "Found NCCL, but the NCCL version is either not 2+ or not "
                   "determinable, will not compile with NCCL distributed "
                   "backend")
  ENDIF()
ELSE()
  MESSAGE(STATUS "Not able to find NCCL, will not "
                 "compile with NCCL distributed backend")
ENDIF()

ADD_DEFINITIONS(-D_THD_CORE=1)

# Can be compiled standalone
IF(NOT THD_INSTALL_BIN_DIR OR NOT THD_INSTALL_LIB_DIR OR NOT THD_INSTALL_INCLUDE_DIR)
  SET(THD_INSTALL_BIN_DIR "bin" CACHE PATH "THD install binary subdirectory")
  SET(THD_INSTALL_LIB_DIR "lib" CACHE PATH "THD install library subdirectory")
  SET(THD_INSTALL_INCLUDE_DIR "include" CACHE PATH "THD install include subdirectory")
ENDIF()

FILE(GLOB_RECURSE master_worker_h RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "master_worker/*.h")
FILE(GLOB_RECURSE master_worker_cpp "master_worker/*.cpp")
FILE(GLOB_RECURSE process_group_h RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "process_group/*.h")
FILE(GLOB_RECURSE process_group_cpp "process_group/*.cpp")
FILE(GLOB_RECURSE base_h RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "base/*.h")
FILE(GLOB_RECURSE base_cpp "base/*.cpp")
FILE(GLOB_RECURSE test_cpp "test/*.cpp")

IF(NOT MPI_FOUND)
  LIST(REMOVE_ITEM base_cpp "${CMAKE_CURRENT_SOURCE_DIR}/base/data_channels/DataChannelMPI.cpp")
  LIST(REMOVE_ITEM test_cpp "${CMAKE_CURRENT_SOURCE_DIR}/test/data_channel_mpi_smoke.cpp")
ENDIF()

IF(NOT GLOO_FOUND)
  LIST(REMOVE_ITEM base_cpp "${CMAKE_CURRENT_SOURCE_DIR}/base/data_channels/DataChannelGloo.cpp")
  LIST(REMOVE_ITEM base_cpp "${CMAKE_CURRENT_SOURCE_DIR}/base/data_channels/Store.cpp")
  LIST(REMOVE_ITEM test_cpp "${CMAKE_CURRENT_SOURCE_DIR}/test/data_channel_gloo_store.cpp")
  LIST(REMOVE_ITEM test_cpp "${CMAKE_CURRENT_SOURCE_DIR}/test/data_channel_gloo_cache.cpp")
ENDIF()

IF(NOT DISTRIBUTED_NCCL_FOUND)
  LIST(REMOVE_ITEM base_cpp "${CMAKE_CURRENT_SOURCE_DIR}/base/data_channels/DataChannelNccl.cpp")
ENDIF()

EXCLUDE_DIR(master_worker_cpp ".*/dispatch/.*\\.cpp$")

SET(all_cpp ${base_cpp} ${process_group_cpp})
SET(all_h THD.h ${base_h} ${process_group_h})

IF(WITH_DISTRIBUTED_MW)
  ADD_DEFINITIONS(-DWITH_DISTRIBUTED_MW=1)
  SET(all_cpp ${all_cpp} ${master_worker_cpp})
  SET(all_h THD.h ${all_h} ${master_worker_h})
ENDIF()

EXCLUDE_DIR(all_cpp ".*/generic/.*\\.cpp$")

# Need to include external NCCL first
IF(DISTRIBUTED_NCCL_FOUND)
  INCLUDE_DIRECTORIES(${NCCL_INCLUDE_DIR})
  FILE(APPEND "${CMAKE_INSTALL_PREFIX}/THD_deps.txt" "${NCCL_LIBRARIES};")
ENDIF()

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

ADD_LIBRARY(THD STATIC ${all_cpp})
set_property(TARGET THD PROPERTY POSITION_INDEPENDENT_CODE ON)

FILE(WRITE "${CMAKE_INSTALL_PREFIX}/THD_deps.txt" "${ATEN_LIBRARIES};")

IF(MPI_FOUND)
  INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
  FILE(APPEND "${CMAKE_INSTALL_PREFIX}/THD_deps.txt" "${MPI_LIBRARIES};")

  IF(MPI_COMPILE_FLAGS)
    MESSAGE(STATUS "MPI_COMPILE_FLAGS: ${MPI_COMPILE_FLAGS}")
    SET_TARGET_PROPERTIES(THD PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  ENDIF()

  IF(MPI_LINK_FLAGS)
    MESSAGE(STATUS "MPI_LINK_FLAGS: ${MPI_LINK_FLAGS}")
    SET_TARGET_PROPERTIES(THD PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
  ENDIF()
ENDIF()

IF(GLOO_FOUND)
  INCLUDE_DIRECTORIES(${GLOO_INCLUDE_DIR})
  FILE(APPEND "${CMAKE_INSTALL_PREFIX}/THD_deps.txt" "${GLOO_LIBRARIES};")
ENDIF()

# Test executables
IF(THD_WITH_TESTS)
  ENABLE_TESTING()
  FIND_PACKAGE(Threads)
  FOREACH(test_source_file ${test_cpp})
    # Prepare test names
    GET_FILENAME_COMPONENT(test_source_file ${test_source_file} NAME)
    STRING(REPLACE ".cpp" "" test_name ${test_source_file})
    SET(test_executable_name "test_${test_name}")

    ADD_EXECUTABLE(${test_executable_name} "test/${test_source_file}")
    TARGET_LINK_LIBRARIES(${test_executable_name} THD ${ATEN_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
    SET_PROPERTY(TARGET ${test_executable_name} PROPERTY CXX_STANDARD 11)
    ADD_TEST(${test_name} ${test_executable_name})
  ENDFOREACH()
ENDIF()

INSTALL(TARGETS THD
  RUNTIME DESTINATION "${THD_INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${THD_INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${THD_INSTALL_LIB_DIR}")

FOREACH(HEADER ${all_h})
  STRING(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
  INSTALL(FILES ${HEADER} DESTINATION ${THD_INSTALL_INCLUDE_DIR}/THD/${DIR})
ENDFOREACH()
