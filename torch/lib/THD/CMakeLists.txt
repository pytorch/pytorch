CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# Can be compiled standalone
IF(NOT THD_INSTALL_BIN_DIR OR NOT THD_INSTALL_LIB_DIR OR NOT THD_INSTALL_INCLUDE_DIR)
  SET(THD_INSTALL_BIN_DIR "bin" CACHE PATH "THD install binary subdirectory")
  SET(THD_INSTALL_LIB_DIR "lib" CACHE PATH "THD install library subdirectory")
  SET(THD_INSTALL_INCLUDE_DIR "include" CACHE PATH "THD install include subdirectory")
ENDIF()

FILE(GLOB_RECURSE master_h "master/*.h")
FILE(GLOB_RECURSE master_cpp "master/*.cpp")
FILE(GLOB_RECURSE worker_h "worker/*.h")
FILE(GLOB_RECURSE worker_cpp "worker/*.cpp")
FILE(GLOB_RECURSE common_h "common/*.h")
FILE(GLOB_RECURSE common_cpp "common/*.cpp")

SET(src_cpp ${master_cpp} ${worker_cpp} ${common_cpp})

ADD_LIBRARY(THD SHARED ${src_cpp})
SET_PROPERTY(TARGET THD PROPERTY CXX_STANDARD 11)

INSTALL(TARGETS THD
  RUNTIME DESTINATION "${THD_INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${THD_INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${THD_INSTALL_LIB_DIR}")

INSTALL(FILES ${master_h} DESTINATION "${THD_INSTALL_INCLUDE_DIR}/THD")

