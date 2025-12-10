if(NOT DEFINED _CMAKE_PROCESSING_LANGUAGE OR _CMAKE_PROCESSING_LANGUAGE STREQUAL "")
  message(FATAL_ERROR "Internal error: _CMAKE_PROCESSING_LANGUAGE is not set")
endif()

# Ubuntu:
# * /usr/bin/llvm-ar-9
# * /usr/bin/llvm-ranlib-9
string(REGEX MATCH "^([0-9]+)" __version_x
    "${CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER_VERSION}")

# Debian:
# * /usr/bin/llvm-ar-4.0
# * /usr/bin/llvm-ranlib-4.0
string(REGEX MATCH "^([0-9]+\\.[0-9]+)" __version_x_y
    "${CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER_VERSION}")

# Try to find tools in the IntelLLVM Clang tools directory
get_filename_component(__intel_llvm_hint_1 "${CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER}" DIRECTORY)
get_filename_component(__intel_llvm_hint_1 "${__intel_llvm_hint_1}/../bin-llvm" REALPATH)

get_filename_component(__intel_llvm_hint_2 "${CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER}" DIRECTORY)
get_filename_component(__intel_llvm_hint_2 "${__intel_llvm_hint_2}/compiler" REALPATH)

get_filename_component(__intel_llvm_hint_3 "${CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER}" DIRECTORY)

set(__intel_llvm_hints ${__intel_llvm_hint_1} ${__intel_llvm_hint_2} ${__intel_llvm_hint_3})

# http://manpages.ubuntu.com/manpages/precise/en/man1/llvm-ar.1.html
find_program(CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER_AR NAMES
    "${_CMAKE_TOOLCHAIN_PREFIX}llvm-ar-${__version_x_y}"
    "${_CMAKE_TOOLCHAIN_PREFIX}llvm-ar-${__version_x}"
    "${_CMAKE_TOOLCHAIN_PREFIX}llvm-ar"
    HINTS ${__intel_llvm_hints}
    NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH
    DOC "LLVM archiver"
)
mark_as_advanced(CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER_AR)

# http://manpages.ubuntu.com/manpages/precise/en/man1/llvm-ranlib.1.html
find_program(CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER_RANLIB NAMES
    "${_CMAKE_TOOLCHAIN_PREFIX}llvm-ranlib-${__version_x_y}"
    "${_CMAKE_TOOLCHAIN_PREFIX}llvm-ranlib-${__version_x}"
    "${_CMAKE_TOOLCHAIN_PREFIX}llvm-ranlib"
    HINTS ${__intel_llvm_hints}
    NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH
    DOC "Generate index for LLVM archive"
)
mark_as_advanced(CMAKE_${_CMAKE_PROCESSING_LANGUAGE}_COMPILER_RANLIB)
