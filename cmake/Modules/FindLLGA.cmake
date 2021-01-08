# - Try to find LLGA

################################################################################
# NOTE: LLGA would not be a standalone repo
#       This CMAKE file should not be included in the final PR
#       All LLGA source code should live in third_party/ideep/mkl-dnn
################################################################################

IF (NOT LLGA_FOUND)

SET(LLGA_LIBRARIES)

SET(LLGA_ROOT "${PROJECT_SOURCE_DIR}/third_party/llga")

ADD_SUBDIRECTORY(${LLGA_ROOT})
IF(NOT TARGET dnnl_graph)
  MESSAGE("Failed to include LLGA target")
  RETURN()
ENDIF(NOT TARGET dnnl_graph)

IF(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)
  TARGET_COMPILE_OPTIONS(dnnl_graph PRIVATE -Wno-maybe-uninitialized)
  TARGET_COMPILE_OPTIONS(dnnl_graph PRIVATE -Wno-strict-overflow)
  TARGET_COMPILE_OPTIONS(dnnl_graph PRIVATE -Wno-error=strict-overflow)
ENDIF(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)
LIST(APPEND LLGA_LIBRARIES dnnl_graph)

SET(LLGA_FOUND TRUE)
MESSAGE(STATUS "Found LLGA: TRUE")

ENDIF(NOT LLGA_FOUND)

