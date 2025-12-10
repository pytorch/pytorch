# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


if(CMAKE_BINARY_DIR)
  message(FATAL_ERROR "CPackFreeBSD.cmake may only be used by CPack internally.")
endif()

if(NOT UNIX)
  message(FATAL_ERROR "CPackFreeBSD.cmake may only be used under UNIX.")
endif()


###
#
# These bits are copied from the Debian packaging file; slightly modified.
# They are used for filling in FreeBSD-packaging variables that can take
# on values from elsewhere -- e.g. the package description may as well be
# copied from Debian.
#
function(_cpack_freebsd_fallback_var OUTPUT_VAR_NAME)
  set(FALLBACK_VAR_NAMES ${ARGN})

  set(VALUE "${${OUTPUT_VAR_NAME}}")
  if(VALUE)
    return()
  endif()

  foreach(variable_name IN LISTS FALLBACK_VAR_NAMES)
    if(${variable_name})
      set(${OUTPUT_VAR_NAME} "${${variable_name}}" PARENT_SCOPE)
      set(VALUE "${${variable_name}}")
      break()
    endif()
  endforeach()
  if(NOT VALUE)
    message(WARNING "Variable ${OUTPUT_VAR_NAME} could not be given a fallback value from (any of) ${FALLBACK_VAR_NAMES}.")
  endif()
endfunction()

function(check_required_var VAR_NAME)
  if(NOT ${VAR_NAME})
    message(FATAL_ERROR "Variable ${VAR_NAME} is not set.")
  endif()
endfunction()

set(_cpack_freebsd_fallback_origin "misc/bogus")

_cpack_freebsd_fallback_var("CPACK_FREEBSD_PACKAGE_NAME"
    "CPACK_PACKAGE_NAME"
    "CMAKE_PROJECT_NAME"
    )

set(_cpack_freebsd_fallback_www "http://example.com/?pkg=${CPACK_FREEBSD_PACKAGE_NAME}")

_cpack_freebsd_fallback_var("CPACK_FREEBSD_PACKAGE_COMMENT"
    "CPACK_PACKAGE_DESCRIPTION_SUMMARY"
    )

# TODO: maybe read the PACKAGE_DESCRIPTION file for the longer
#       FreeBSD pkg-descr?
_cpack_freebsd_fallback_var("CPACK_FREEBSD_PACKAGE_DESCRIPTION"
    "CPACK_DEBIAN_PACKAGE_DESCRIPTION"
    "CPACK_PACKAGE_DESCRIPTION_SUMMARY"
    "PACKAGE_DESCRIPTION"
    )

# There's really only one homepage for a project, so
# reuse the Debian setting if it's there.
_cpack_freebsd_fallback_var("CPACK_FREEBSD_PACKAGE_WWW"
    "CPACK_PACKAGE_HOMEPAGE_URL"
    "CPACK_DEBIAN_PACKAGE_HOMEPAGE"
    "_cpack_freebsd_fallback_www"
    )

_cpack_freebsd_fallback_var("CPACK_FREEBSD_PACKAGE_VERSION"
    "CMAKE_PROJECT_VERSION"
    "${CMAKE_PROJECT_NAME}_VERSION"
    "PROJECT_VERSION"
    "CPACK_PACKAGE_VERSION"
    "CPACK_PACKAGE_VERSION"
    )

_cpack_freebsd_fallback_var("CPACK_FREEBSD_PACKAGE_MAINTAINER"
    "CPACK_PACKAGE_CONTACT"
    )

_cpack_freebsd_fallback_var("CPACK_FREEBSD_PACKAGE_LICENSE"
    "CPACK_RPM_PACKAGE_LICENSE"
    )

_cpack_freebsd_fallback_var("CPACK_FREEBSD_PACKAGE_ORIGIN"
  "_cpack_freebsd_fallback_origin"
  )

if(NOT CPACK_FREEBSD_PACKAGE_CATEGORIES)
  string(REGEX REPLACE "/.*" "" CPACK_FREEBSD_PACKAGE_CATEGORIES ${CPACK_FREEBSD_PACKAGE_ORIGIN})
endif()

check_required_var("CPACK_FREEBSD_PACKAGE_NAME")
check_required_var("CPACK_FREEBSD_PACKAGE_ORIGIN")
check_required_var("CPACK_FREEBSD_PACKAGE_VERSION")
check_required_var("CPACK_FREEBSD_PACKAGE_MAINTAINER")
check_required_var("CPACK_FREEBSD_PACKAGE_COMMENT")
check_required_var("CPACK_FREEBSD_PACKAGE_DESCRIPTION")
check_required_var("CPACK_FREEBSD_PACKAGE_WWW")
check_required_var("CPACK_FREEBSD_PACKAGE_LICENSE")
