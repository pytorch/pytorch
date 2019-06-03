# - Find the Catch test framework or download it (single header)
#
# This is a quick module for internal use. It assumes that Catch is
# REQUIRED and that a minimum version is provided (not EXACT). If
# a suitable version isn't found locally, the single header file
# will be downloaded and placed in the build dir: PROJECT_BINARY_DIR.
#
# This code sets the following variables:
#  CATCH_INCLUDE_DIR      - path to catch.hpp
#  CATCH_VERSION          - version number

if(NOT Catch_FIND_VERSION)
  message(FATAL_ERROR "A version number must be specified.")
elseif(Catch_FIND_REQUIRED)
  message(FATAL_ERROR "This module assumes Catch is not required.")
elseif(Catch_FIND_VERSION_EXACT)
  message(FATAL_ERROR "Exact version numbers are not supported, only minimum.")
endif()

# Extract the version number from catch.hpp
function(_get_catch_version)
  file(STRINGS "${CATCH_INCLUDE_DIR}/catch.hpp" version_line REGEX "Catch v.*" LIMIT_COUNT 1)
  if(version_line MATCHES "Catch v([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    set(CATCH_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}" PARENT_SCOPE)
  endif()
endfunction()

# Download the single-header version of Catch
function(_download_catch version destination_dir)
  message(STATUS "Downloading catch v${version}...")
  set(url https://github.com/philsquared/Catch/releases/download/v${version}/catch.hpp)
  file(DOWNLOAD ${url} "${destination_dir}/catch.hpp" STATUS status)
  list(GET status 0 error)
  if(error)
    message(FATAL_ERROR "Could not download ${url}")
  endif()
  set(CATCH_INCLUDE_DIR "${destination_dir}" CACHE INTERNAL "")
endfunction()

# Look for catch locally
find_path(CATCH_INCLUDE_DIR NAMES catch.hpp PATH_SUFFIXES catch)
if(CATCH_INCLUDE_DIR)
  _get_catch_version()
endif()

# Download the header if it wasn't found or if it's outdated
if(NOT CATCH_VERSION OR CATCH_VERSION VERSION_LESS ${Catch_FIND_VERSION})
  if(DOWNLOAD_CATCH)
    _download_catch(${Catch_FIND_VERSION} "${PROJECT_BINARY_DIR}/catch/")
    _get_catch_version()
  else()
    set(CATCH_FOUND FALSE)
    return()
  endif()
endif()

set(CATCH_FOUND TRUE)
