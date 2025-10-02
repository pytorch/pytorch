# This is a wrapper of the upstream `./upstream/FindCUDA.cmake` that
# automatically includes `./upstream/CMakeInitializeConfigs.cmake` before
# `./upstream/FindCUDA.cmake`. The `CMakeInitializeConfigs.cmake`, which is
# absent in old CMake versions, creates some necessary variables for the later
# to run.
# See ./README.md for details.

set(UPSTREAM_FIND_CUDA_DIR "${CMAKE_CURRENT_LIST_DIR}/upstream/")

include("${UPSTREAM_FIND_CUDA_DIR}/FindCUDA.cmake")
