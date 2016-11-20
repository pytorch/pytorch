CMAKE_MINIMUM_REQUIRED(VERSION 2.6)


FIND_PACKAGE(MPI)

IF(MPI_FOUND)
  ADD_DEFINITIONS(-DWITH_MPI=1)
ENDIF()

# Can be compiled standalone
IF(NOT THD_INSTALL_BIN_DIR OR NOT THD_INSTALL_LIB_DIR OR NOT THD_INSTALL_INCLUDE_DIR)
  SET(THD_INSTALL_BIN_DIR "bin" CACHE PATH "THD install binary subdirectory")
  SET(THD_INSTALL_LIB_DIR "lib" CACHE PATH "THD install library subdirectory")
  SET(THD_INSTALL_INCLUDE_DIR "include" CACHE PATH "THD install include subdirectory")
ENDIF()

FILE(GLOB_RECURSE master_worker_h "master_worker/*.h")
FILE(GLOB_RECURSE master_worker_cpp "master_worker/*.cpp")
FILE(GLOB_RECURSE base_h "base/*.h")
FILE(GLOB_RECURSE base_cpp "base/*.cpp")
FILE(GLOB_RECURSE test_cpp "test/*.cpp")

IF(NOT MPI_FOUND)
  LIST(REMOVE_ITEM base_cpp "${CMAKE_CURRENT_SOURCE_DIR}/common/DataChannelMPI.cpp")
  LIST(REMOVE_ITEM test_cpp "${CMAKE_CURRENT_SOURCE_DIR}/test/data_channel_mpi_smoke.cpp")
ENDIF()

SET(all_cpp ${base_cpp} ${master_worker_cpp})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
ADD_LIBRARY(THD SHARED ${all_cpp})
SET_PROPERTY(TARGET THD PROPERTY CXX_STANDARD 11)

IF(MPI_FOUND)
  INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
  TARGET_LINK_LIBRARIES(THD ${MPI_LIBRARIES})

  IF(MPI_COMPILE_FLAGS)
    SET_TARGET_PROPERTIES(THD PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  ENDIF()

  IF(MPI_LINK_FLAGS)
    SET_TARGET_PROPERTIES(THD PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
  ENDIF()
ENDIF()

# Test executables
ENABLE_TESTING()

FOREACH(test_source_file ${test_cpp})
  # Prepare test names
  GET_FILENAME_COMPONENT(test_source_file ${test_source_file} NAME)
  STRING(REPLACE ".cpp" "" test_name ${test_source_file})
  SET(test_executable_name "test_${test_name}")

  ADD_EXECUTABLE(${test_executable_name} "test/${test_source_file}")
  TARGET_LINK_LIBRARIES(${test_executable_name} THD)
  SET_PROPERTY(TARGET ${test_executable_name} PROPERTY CXX_STANDARD 11)
  ADD_TEST(${test_name} ${test_executable_name})
ENDFOREACH()

INSTALL(TARGETS THD
  RUNTIME DESTINATION "${THD_INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${THD_INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${THD_INSTALL_LIB_DIR}")

INSTALL(FILES ${master_h} DESTINATION "${THD_INSTALL_INCLUDE_DIR}/THD")
