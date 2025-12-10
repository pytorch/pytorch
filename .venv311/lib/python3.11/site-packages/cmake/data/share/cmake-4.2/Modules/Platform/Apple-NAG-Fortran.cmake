# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

set(CMAKE_Fortran_VERBOSE_FLAG "-Wl,-v") # Runs gcc under the hood.

# FIXME(#25900): We need -fpp explicitly on case-insensitive filesystems,
# but this does not work with the Ninja generator's separate preprocessing
# and compilation steps.
if(NOT CMAKE_GENERATOR MATCHES "^Ninja")
  set(CMAKE_Fortran_COMPILE_OBJECT
    "<CMAKE_Fortran_COMPILER> -fpp -o <OBJECT> <DEFINES> <INCLUDES> <FLAGS> -c <SOURCE>")
endif()

set(CMAKE_Fortran_OSX_COMPATIBILITY_VERSION_FLAG "-Wl,-compatibility_version -Wl,")
set(CMAKE_Fortran_OSX_CURRENT_VERSION_FLAG "-Wl,-current_version -Wl,")
set(CMAKE_SHARED_MODULE_CREATE_Fortran_FLAGS "-Wl,-bundle")
set(CMAKE_SHARED_LIBRARY_CREATE_Fortran_FLAGS "-Wl,-shared")
set(CMAKE_SHARED_LIBRARY_SONAME_Fortran_FLAG "-Wl,-install_name -Wl,")
set(CMAKE_Fortran_CREATE_SHARED_LIBRARY
  "<CMAKE_Fortran_COMPILER> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> -o <TARGET> <SONAME_FLAG><TARGET_INSTALLNAME_DIR><TARGET_SONAME> <OBJECTS> <LINK_LIBRARIES>")
