CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
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

IF(NOT Torch_FOUND)
  FIND_PACKAGE(Torch REQUIRED)
ENDIF()

# Can be compiled standalone
IF(NOT THPP_INSTALL_BIN_DIR OR NOT THPP_INSTALL_LIB_DIR OR NOT THPP_INSTALL_INCLUDE_DIR)
  SET(THPP_INSTALL_BIN_DIR "bin" CACHE PATH "THPP install binary subdirectory")
  SET(THPP_INSTALL_LIB_DIR "lib" CACHE PATH "THPP install library subdirectory")
  SET(THPP_INSTALL_INCLUDE_DIR "include" CACHE PATH "THPP install include subdirectory")
ENDIF()

FILE(GLOB_RECURSE base_h RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h" "*.hpp")
FILE(GLOB_RECURSE base_cpp "*.cpp")

SET(all_cpp ${base_cpp})
SET(all_h ${base_h})

EXCLUDE_DIR(all_cpp ".*/generic/.*\\.cpp$")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
ADD_LIBRARY(THPP SHARED ${all_cpp})
SET_PROPERTY(TARGET THPP PROPERTY CXX_STANDARD 11)
SET_TARGET_PROPERTIES(THPP PROPERTIES VERSION 1 SOVERSION 1)

TARGET_LINK_LIBRARIES(THPP PRIVATE ${TH_LIBRARIES})

# Test executables
# ENABLE_TESTING()

# FIND_PACKAGE(Threads)
# FOREACH(test_source_file ${test_cpp})
#   # Prepare test names
#   GET_FILENAME_COMPONENT(test_source_file ${test_source_file} NAME)
#   STRING(REPLACE ".cpp" "" test_name ${test_source_file})
#   SET(test_executable_name "test_${test_name}")
#
#   ADD_EXECUTABLE(${test_executable_name} "test/${test_source_file}")
#   TARGET_LINK_LIBRARIES(${test_executable_name} THPP TH ${CMAKE_THREAD_LIBS_INIT})
#   SET_PROPERTY(TARGET ${test_executable_name} PROPERTY CXX_STANDARD 11)
#   ADD_TEST(${test_name} ${test_executable_name})
# ENDFOREACH()

INSTALL(TARGETS THPP
  RUNTIME DESTINATION "${THPP_INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${THPP_INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${THPP_INSTALL_LIB_DIR}")

FOREACH(HEADER ${all_h})
  STRING(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
  INSTALL(FILES ${HEADER} DESTINATION ${THPP_INSTALL_INCLUDE_DIR}/THPP/${DIR})
ENDFOREACH()
