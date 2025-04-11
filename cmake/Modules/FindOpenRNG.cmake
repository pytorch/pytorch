# Build with OpenRNG for the Arm architecture
#
# Note: The OpenRNG library is available at:
#   https://git.gitlab.arm.com/libraries/openrng.git
#   and must be built separately.
#
# To build OpenRNG as a shared library, use:
#   cmake -DBUILD_SHARED_LIB=ON
# For more details, refer to: https://gitlab.arm.com/libraries/openrng/-/issues/4
#
# To integrate OpenRNG into this build dynamically, set the environment variable:
#   OPENRNG_ROOT_DIR
# This variable should point to the location of the OpenRNG build directory.
#
# The FindOpenRNG.cmake script will verify the specified path during configuration.

find_path(
  OPENRNG_INCLUDE_DIR openrng.h
  PATHS
  $ENV{OPENRNG_ROOT_DIR}/install/include
  /usr/local/include
  /usr/include
)
find_library(
  OPENRNG_LIBRARY NAMES openrng
  PATHS
  $ENV{OPENRNG_ROOT_DIR}/install/lib
  /usr/local/lib
  /usr/lib
)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  OpenRNG DEFAULT_MSG OPENRNG_INCLUDE_DIR OPENRNG_LIBRARY)
if(OpenRNG_FOUND)
  message(
    STATUS
    "Found OpenRNG  (include: ${OPENRNG_INCLUDE_DIR}, library: ${OPENRNG_LIBRARY})")
  add_library(openrng SHARED IMPORTED)
  mark_as_advanced(OPENRNG_INCLUDE_DIR OPENRNG_LIBRARY)
endif()