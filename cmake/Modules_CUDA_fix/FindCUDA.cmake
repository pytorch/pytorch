# This is a wrapper of the upstream `./upstream/FindCUDA.cmake`.
# See ./README.md for details.

set(UPSTREAM_FIND_CUDA_DIR "${CMAKE_CURRENT_LIST_DIR}/upstream/")

include("${UPSTREAM_FIND_CUDA_DIR}/FindCUDA.cmake")
