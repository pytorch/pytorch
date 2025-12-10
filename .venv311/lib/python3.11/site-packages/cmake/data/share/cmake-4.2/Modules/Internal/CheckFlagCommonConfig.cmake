# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# Do NOT include this module directly into any of your code. It is meant as
# a library for Check*CompilerFlag.cmake and Check*LinkerFlag.cma modules.
# It's content may change in any way between releases.

include_guard(GLOBAL)

macro(CMAKE_CHECK_FLAG_COMMON_INIT _FUNC _LANG _SRC _PATTERNS)
  if("${_LANG}" STREQUAL "C")
    set(${_SRC} "int main(void) { return 0; }")
    set(${_PATTERNS}
      FAIL_REGEX "command[ -]line option [^\n]* is valid for [^\n]* but not for C"
      FAIL_REGEX "-Werror=[^\n]* argument [^\n]* is not valid for C"
    )
  elseif("${_LANG}" STREQUAL "CXX")
    set(${_SRC} "int main() { return 0; }")
    set(${_PATTERNS}
      FAIL_REGEX "command[ -]line option [^\n]* is valid for [^\n]* but not for C\\+\\+"
      FAIL_REGEX "-Werror=[^\n]* argument [^\n]* is not valid for C\\+\\+"
    )
  elseif("${_LANG}" STREQUAL "CUDA")
    set(${_SRC} "__host__ int main() { return 0; }")
    set(${_PATTERNS}
      FAIL_REGEX "command[ -]line option [^\n]* is valid for [^\n]* but not for C\\+\\+" # Host GNU
      FAIL_REGEX "argument unused during compilation: [^\n]*" # Clang
    )
  elseif("${_LANG}" STREQUAL "Fortran")
    set(${_SRC} "       program test\n       stop\n       end program")
    set(${_PATTERNS}
      FAIL_REGEX "command[ -]line option [^\n]* is valid for [^\n]* but not for Fortran"
      FAIL_REGEX "argument unused during compilation: [^\n]*" # LLVMFlang
    )
  elseif("${_LANG}" STREQUAL "HIP")
    set(${_SRC} "__host__ int main() { return 0; }")
    set(${_PATTERNS}
      FAIL_REGEX "argument unused during compilation: [^\n]*" # Clang
    )
  elseif("${_LANG}" STREQUAL "OBJC")
    set(${_SRC} [=[
      #ifndef __OBJC__
      #  error "Not an Objective-C compiler"
      #endif
      int main(void) { return 0; }]=])
    set(${_PATTERNS}
      FAIL_REGEX "command[ -]line option [^\n]* is valid for [^\n]* but not for Objective-C" # GNU
      FAIL_REGEX "argument unused during compilation: [^\n]*" # Clang
    )
  elseif("${_LANG}" STREQUAL "OBJCXX")
    set(${_SRC} [=[
      #ifndef __OBJC__
      #  error "Not an Objective-C++ compiler"
      #endif
      int main(void) { return 0; }]=])
    set(${_PATTERNS}
      FAIL_REGEX "command[ -]line option [^\n]* is valid for [^\n]* but not for Objective-C\\+\\+" # GNU
      FAIL_REGEX "argument unused during compilation: [^\n]*" # Clang
    )
  elseif("${_LANG}" STREQUAL "ISPC")
    set(${_SRC} "float func(uniform int32, float a) { return a / 2.25; }")
  elseif("${_LANG}" STREQUAL "Swift")
    set(${_SRC} "func blarpy() { }")
  else()
    message (SEND_ERROR "${_FUNC}: ${_LANG}: unknown language.")
    return()
  endif()

  get_property (_supported_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if (NOT "${_LANG}" IN_LIST _supported_languages)
    message (SEND_ERROR "${_FUNC}: ${_LANG}: needs to be enabled before use.")
    return()
  endif()
  # Normalize locale during test compilation.
  set(_CFCC_locale_vars LC_ALL LC_MESSAGES LANG)
  foreach(v IN LISTS _CFCC_locale_vars)
    set(_CMAKE_CHECK_FLAG_COMMON_CONFIG_locale_vars_saved_${v} "$ENV{${v}}")
    set(ENV{${v}} C)
  endforeach()
endmacro()

macro(CMAKE_CHECK_FLAG_COMMON_FINISH)
  foreach(v IN LISTS _CFCC_locale_vars)
    set(ENV{${v}} ${_CMAKE_CHECK_FLAG_COMMON_CONFIG_locale_vars_saved_${v}})
  endforeach()
endmacro()
