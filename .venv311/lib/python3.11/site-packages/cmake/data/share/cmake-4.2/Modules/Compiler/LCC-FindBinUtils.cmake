if(NOT DEFINED _CMAKE_PROCESSING_LANGUAGE OR _CMAKE_PROCESSING_LANGUAGE STREQUAL "")
  message(FATAL_ERROR "Internal error: _CMAKE_PROCESSING_LANGUAGE is not set")
endif()

# Ubuntu 16.04:
# * /usr/bin/gcc-ar-5
# * /usr/bin/gcc-ranlib-5
string(REGEX MATCH "^([0-9]+)" __version_x
    "${CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER_VERSION}")

string(REGEX MATCH "^([0-9]+\\.[0-9]+)" __version_x_y
    "${CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER_VERSION}")

# Try to find tools in the same directory as GCC itself
get_filename_component(__gcc_hints "${CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER}" DIRECTORY)

# http://manpages.ubuntu.com/manpages/wily/en/man1/gcc-ar.1.html
find_program(CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER_AR NAMES
    "${_CMAKE_TOOLCHAIN_PREFIX}gcc-ar-${__version_x_y}"
    "${_CMAKE_TOOLCHAIN_PREFIX}gcc-ar-${__version_x}"
    "${_CMAKE_TOOLCHAIN_PREFIX}gcc-ar${_CMAKE_COMPILER_SUFFIX}"
    HINTS ${__gcc_hints}
    NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH
    DOC "A wrapper around 'ar' adding the appropriate '--plugin' option for the GCC compiler"
)
mark_as_advanced(CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER_AR)

# http://manpages.ubuntu.com/manpages/wily/en/man1/gcc-ranlib.1.html
find_program(CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER_RANLIB NAMES
    "${_CMAKE_TOOLCHAIN_PREFIX}gcc-ranlib-${__version_x_y}"
    "${_CMAKE_TOOLCHAIN_PREFIX}gcc-ranlib-${__version_x}"
    "${_CMAKE_TOOLCHAIN_PREFIX}gcc-ranlib${_CMAKE_COMPILER_SUFFIX}"
    HINTS ${__gcc_hints}
    NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH
    DOC "A wrapper around 'ranlib' adding the appropriate '--plugin' option for the GCC compiler"
)
mark_as_advanced(CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER_RANLIB)
