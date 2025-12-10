# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include(Compiler/MSVC)
__compiler_msvc(C)

include(Compiler/CMakeCommonCompilerMacros)

if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 19.27)
  set(CMAKE_C90_STANDARD_COMPILE_OPTION "")
  set(CMAKE_C90_EXTENSION_COMPILE_OPTION "")
  set(CMAKE_C99_STANDARD_COMPILE_OPTION "")
  set(CMAKE_C99_EXTENSION_COMPILE_OPTION "")
  set(CMAKE_C11_STANDARD_COMPILE_OPTION "-std:c11")
  set(CMAKE_C11_EXTENSION_COMPILE_OPTION "-std:c11")

  set(CMAKE_C_STANDARD_LATEST 11)

  if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 19.28)
    set(CMAKE_C90_STANDARD__HAS_FULL_SUPPORT ON)
    set(CMAKE_C99_STANDARD__HAS_FULL_SUPPORT ON)
    set(CMAKE_C11_STANDARD__HAS_FULL_SUPPORT ON)
    set(CMAKE_C17_STANDARD_COMPILE_OPTION "-std:c17")
    set(CMAKE_C17_EXTENSION_COMPILE_OPTION "-std:c17")
    set(CMAKE_C_STANDARD_LATEST 17)
  else()
    # Special case for 19.27 (VS 16.7): C11 has partial support.
    macro(cmake_record_c_compile_features)
      _has_compiler_features_c(90)
      _has_compiler_features_c(99)
      list(APPEND CMAKE_C11_COMPILE_FEATURES c_std_11)
      set(_result 0) # expected by cmake_determine_compile_features
    endmacro()
  endif()

  if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 19.39)
    # VS 17.10 did not have a "stdclatest" value for LanguageStandard_C.
    if(NOT CMAKE_GENERATOR MATCHES "Visual Studio"
        OR CMAKE_VS_VERSION_BUILD_NUMBER VERSION_GREATER_EQUAL 17.11)
      set(CMAKE_C23_STANDARD_COMPILE_OPTION "-std:clatest")
      set(CMAKE_C23_EXTENSION_COMPILE_OPTION "-std:clatest")
      set(CMAKE_C_STANDARD_LATEST 23)
    endif()
  endif()

  __compiler_check_default_language_standard(C 19.27 99)
else()
  # MSVC has no specific options to set C language standards, but set them as
  # empty strings anyways so the feature test infrastructure can at least check
  # to see if they are defined.
  set(CMAKE_C90_STANDARD_COMPILE_OPTION "")
  set(CMAKE_C90_EXTENSION_COMPILE_OPTION "")
  set(CMAKE_C99_STANDARD_COMPILE_OPTION "")
  set(CMAKE_C99_EXTENSION_COMPILE_OPTION "")
  set(CMAKE_C11_STANDARD_COMPILE_OPTION "")
  set(CMAKE_C11_EXTENSION_COMPILE_OPTION "")
  set(CMAKE_C_STANDARD_LATEST 11)

  # There is no meaningful default for this
  set(CMAKE_C_STANDARD_DEFAULT "")

  # There are no C compiler modes so we hard-code the known compiler supported
  # features. Override the default macro for this special case.  Pretend that
  # all language standards are available so that at least compilation
  # can be attempted.
  macro(cmake_record_c_compile_features)
    list(APPEND CMAKE_C_COMPILE_FEATURES
      c_std_90
      c_std_99
      c_std_11
      c_std_17
      c_std_23
      c_function_prototypes
      )
    list(APPEND CMAKE_C90_COMPILE_FEATURES c_std_90 c_function_prototypes)
    list(APPEND CMAKE_C99_COMPILE_FEATURES c_std_99)
    list(APPEND CMAKE_C11_COMPILE_FEATURES c_std_11)
    if (CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 14.0)
      list(APPEND CMAKE_C_COMPILE_FEATURES c_variadic_macros)
      list(APPEND CMAKE_C99_COMPILE_FEATURES c_variadic_macros)
    endif()
    set(_result 0) # expected by cmake_determine_compile_features
  endmacro()
endif()

set(CMAKE_C_COMPILE_OPTIONS_EXPLICIT_LANGUAGE -TC)
