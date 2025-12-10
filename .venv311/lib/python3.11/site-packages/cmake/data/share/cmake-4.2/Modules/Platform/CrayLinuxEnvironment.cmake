# CrayLinuxEnvironment: loaded by users cross-compiling on a Cray front-end
# node by specifying "-DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment" to cmake

if(DEFINED ENV{CRAYOS_VERSION})
  set(CMAKE_SYSTEM_VERSION "$ENV{CRAYOS_VERSION}")
elseif(DEFINED ENV{XTOS_VERSION})
  set(CMAKE_SYSTEM_VERSION "$ENV{XTOS_VERSION}")
elseif(EXISTS /etc/opt/cray/release/cle-release)
  cmake_policy(PUSH)
  cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>
  file(STRINGS /etc/opt/cray/release/cle-release release REGEX "^RELEASE=.*")
  cmake_policy(POP)
  string(REGEX REPLACE "^RELEASE=(.*)$" "\\1" CMAKE_SYSTEM_VERSION "${release}")
  unset(release)
elseif(EXISTS /etc/opt/cray/release/clerelease)
  file(READ /etc/opt/cray/release/clerelease CMAKE_SYSTEM_VERSION)
endif()

# Guard against multiple messages
if(NOT __CrayLinuxEnvironment_message)
  set(__CrayLinuxEnvironment_message 1 CACHE INTERNAL "")
  if(NOT CMAKE_SYSTEM_VERSION)
    message(STATUS "CrayLinuxEnvironment: Unable to determine CLE version.  This platform file should only be used from inside the Cray Linux Environment for targeting compute nodes (NIDs).")
  else()
    message(STATUS "Cray Linux Environment ${CMAKE_SYSTEM_VERSION}")
  endif()
endif()

# All cray systems are x86 CPUs and have been for quite some time
# Note: this may need to change in the future with 64-bit ARM
set(CMAKE_SYSTEM_PROCESSOR "x86_64")

# Don't override shared lib support if it's already been set and possibly
# overridden elsewhere by the CrayPrgEnv module
if(NOT CMAKE_FIND_LIBRARY_SUFFIXES)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".a")
  set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
endif()

# The rest of this file is based on UnixPaths.cmake, adjusted for Cray

# add the install directory of the running cmake to the search directories
# CMAKE_ROOT is CMAKE_INSTALL_PREFIX/share/cmake, so we need to go two levels up
get_filename_component(__cmake_install_dir "${CMAKE_ROOT}" PATH)
get_filename_component(__cmake_install_dir "${__cmake_install_dir}" PATH)

# Note: Some Cray's have the SYSROOT_DIR variable defined, pointing to a copy
# of the NIDs userland.  If so, then we'll use it.  Otherwise, just assume
# the userland from the login node is ok

# List common installation prefixes.  These will be used for all
# search types.
list(APPEND CMAKE_SYSTEM_PREFIX_PATH
  # Standard
  $ENV{SYSROOT_DIR}/usr/local $ENV{SYSROOT_DIR}/usr $ENV{SYSROOT_DIR}/

  # CMake install location
  "${__cmake_install_dir}"
  )
if (NOT CMAKE_FIND_NO_INSTALL_PREFIX)
  list(APPEND CMAKE_SYSTEM_PREFIX_PATH
    # Project install destination.
    "${CMAKE_INSTALL_PREFIX}"
  )
  if(CMAKE_STAGING_PREFIX)
    list(APPEND CMAKE_SYSTEM_PREFIX_PATH
      # User-supplied staging prefix.
      "${CMAKE_STAGING_PREFIX}"
    )
  endif()
endif()
_cmake_record_install_prefix()

list(APPEND CMAKE_SYSTEM_INCLUDE_PATH
  $ENV{SYSROOT_DIR}/usr/include/X11
)
list(APPEND CMAKE_SYSTEM_LIBRARY_PATH
  $ENV{SYSROOT_DIR}/usr/local/lib64
  $ENV{SYSROOT_DIR}/usr/lib64
  $ENV{SYSROOT_DIR}/lib64
)
list(APPEND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
  $ENV{SYSROOT_DIR}/usr/local/lib64
  $ENV{SYSROOT_DIR}/usr/lib64
  $ENV{SYSROOT_DIR}/lib64
)

# Enable use of lib64 search path variants by default.
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)
