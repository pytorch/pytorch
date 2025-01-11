# Build with Compute Library backend for the Arm architecture
# Note: Compute Library is available from: https://github.com/ARM-software/ComputeLibrary
#   and must be built separately. The location of the Compute Library build
#   must be set with the env var ACL_ROOT_DIR. This path will be checked later
#   as part of FindACL.cmake in oneDNN.

if(NOT USE_ONEDNN_ACL)
    RETURN()
endif()

set(DNNL_AARCH64_USE_ACL ON CACHE BOOL "" FORCE)

# Check the Compute Library version number.
# Note: oneDNN / MKL-DNN v2.2 onwards will check the Compute Library version
#   the version check here can be removed once PyTorch transitions to v2.2.
set(ACL_MINIMUM_VERSION "21.02")

file(GLOB_RECURSE ACL_VERSION_FILE $ENV{ACL_ROOT_DIR}/*/arm_compute_version.embed)

if("${ACL_VERSION_FILE}" STREQUAL "")
  message(WARNING "Build may fail: Could not determine ACL version (minimum required is ${ACL_MINIMUM_VERSION})")
else()
  file(READ ${ACL_VERSION_FILE} ACL_VERSION_STRING)
  string(REGEX MATCH "v([0-9]+\\.[0-9]+)" ACL_VERSION "${ACL_VERSION_STRING}")
  set(ACL_VERSION "${CMAKE_MATCH_1}")

  if("${ACL_VERSION}" VERSION_EQUAL "0.0")
    # Unreleased ACL versions come with version string "v0.0-unreleased", and may not be compatible with oneDNN.
    # It is recommended to use the latest release of ACL.
    message(WARNING "Build may fail: Using unreleased ACL version (minimum required is ${ACL_MINIMUM_VERSION})")
  elseif(${ACL_VERSION} VERSION_LESS ${ACL_MINIMUM_VERSION})
    message(FATAL_ERROR "Detected ACL version ${ACL_VERSION}, but minimum required is ${ACL_MINIMUM_VERSION}")
  endif()
endif()
