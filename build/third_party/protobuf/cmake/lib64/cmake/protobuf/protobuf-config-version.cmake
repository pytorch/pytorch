set(PACKAGE_VERSION "3.6.1")
set(${PACKAGE_FIND_NAME}_VERSION_PRERELEASE "" PARENT_SCOPE)

# Prerelease versions cannot be passed in directly via the find_package command,
# so we allow users to specify it in a variable
if(NOT DEFINED "${PACKAGE_FIND_NAME}_FIND_VERSION_PRERELEASE")
  set("${${PACKAGE_FIND_NAME}_FIND_VERSION_PRERELEASE}" "")
else()
  set(PACKAGE_FIND_VERSION ${PACKAGE_FIND_VERSION}-${${PACKAGE_FIND_NAME}_FIND_VERSION_PRERELEASE})
endif()
set(PACKAGE_FIND_VERSION_PRERELEASE "${${PACKAGE_FIND_NAME}_FIND_VERSION_PRERELEASE}")

# VERSION_EQUAL ignores the prerelease strings, so we use STREQUAL.
if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)
  set(PACKAGE_VERSION_EXACT TRUE)
endif()

set(PACKAGE_VERSION_COMPATIBLE TRUE) #Assume true until shown otherwise

if(PACKAGE_FIND_VERSION) #Only perform version checks if one is given
  if(NOT PACKAGE_FIND_VERSION_MAJOR EQUAL "3")
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
  elseif(PACKAGE_FIND_VERSION VERSION_GREATER PACKAGE_VERSION)
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
  elseif(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)
    # Do not match prerelease versions to non-prerelease version requests.
      if(NOT "" STREQUAL "" AND PACKAGE_FIND_VERSION_PRERELEASE STREQUAL "")
      message(AUTHOR_WARNING "To use this prerelease version of ${PACKAGE_FIND_NAME}, set ${PACKAGE_FIND_NAME}_FIND_VERSION_PRERELEASE to '' or greater.")
      set(PACKAGE_VERSION_COMPATIBLE FALSE)
    endif()

    # Not robustly SemVer compliant, but protobuf never uses '.' separated prerelease identifiers.
    if(PACKAGE_FIND_VERSION_PRERELEASE STRGREATER "")
      set(PACKAGE_VERSION_COMPATIBLE FALSE)
    endif()
  endif()
endif()

# Check and save build options used to create this package
macro(_check_and_save_build_option OPTION VALUE)
  if(DEFINED ${PACKAGE_FIND_NAME}_${OPTION} AND
    NOT ${PACKAGE_FIND_NAME}_${OPTION} STREQUAL ${VALUE})
    set(PACKAGE_VERSION_UNSUITABLE TRUE)
  endif()
  set(${PACKAGE_FIND_NAME}_${OPTION} ${VALUE} PARENT_SCOPE)
endmacro()
_check_and_save_build_option(WITH_ZLIB OFF)
_check_and_save_build_option(MSVC_STATIC_RUNTIME OFF)
_check_and_save_build_option(BUILD_SHARED_LIBS OFF)

# if the installed or the using project don't have CMAKE_SIZEOF_VOID_P set, ignore it:
if(CMAKE_SIZEOF_VOID_P AND "8")
  # check that the installed version has the same 32/64bit-ness as the one which is currently searching:
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL "8")
    math(EXPR installedBits "8 * 8")
    set(PACKAGE_VERSION "${PACKAGE_VERSION} (${installedBits}bit)")
    set(PACKAGE_VERSION_UNSUITABLE TRUE)
  endif()
endif()

