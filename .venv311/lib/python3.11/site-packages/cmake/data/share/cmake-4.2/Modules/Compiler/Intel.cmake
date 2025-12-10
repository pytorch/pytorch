# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__COMPILER_INTEL)
  return()
endif()
set(__COMPILER_INTEL 1)

include(Compiler/CMakeCommonCompilerMacros)

if(CMAKE_HOST_WIN32)
  # MSVC-like
  macro(__compiler_intel lang)
    if("x${lang}" STREQUAL "xFortran")
      set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "-warn:errors")
    else()
      set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "-Werror-all")
    endif()

    set(CMAKE_${lang}_LINK_MODE LINKER)
  endmacro()
else()
  # GNU-like
  macro(__compiler_intel lang)
    set(CMAKE_${lang}_VERBOSE_FLAG "-v")

    string(APPEND CMAKE_${lang}_FLAGS_INIT " ")
    string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -g")
    string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -Os")
    string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " -O3")
    string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " -O2 -g")

    # Compiler + IPO does not recognize --dependency-file link option
    set(CMAKE_${lang}_LINKER_DEPFILE_SUPPORTED FALSE)

    if("${lang}" STREQUAL "CXX")
      set(CMAKE_${lang}_COMPILER_PREDEFINES_COMMAND "${CMAKE_${lang}_COMPILER}")
      if(CMAKE_${lang}_COMPILER_ARG1)
        separate_arguments(_COMPILER_ARGS NATIVE_COMMAND "${CMAKE_${lang}_COMPILER_ARG1}")
        list(APPEND CMAKE_${lang}_COMPILER_PREDEFINES_COMMAND ${_COMPILER_ARGS})
        unset(_COMPILER_ARGS)
      endif()
      list(APPEND CMAKE_${lang}_COMPILER_PREDEFINES_COMMAND "-w" "-QdM" "-P" "-Za" "${CMAKE_ROOT}/Modules/CMakeCXXCompilerABI.cpp")
    endif()

    if("x${lang}" STREQUAL "xFortran")
      set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "-warn" "errors")
    else()
      # Precompile Headers
      set(CMAKE_PCH_EXTENSION .pchi)
      set(CMAKE_LINK_PCH ON)
      set(CMAKE_PCH_EPILOGUE "#pragma hdrstop")
      set(CMAKE_${lang}_COMPILE_OPTIONS_INVALID_PCH -Winvalid-pch)
      set(CMAKE_${lang}_COMPILE_OPTIONS_USE_PCH -Wno-pch-messages -pch-use <PCH_FILE> -include <PCH_HEADER>)
      set(CMAKE_${lang}_COMPILE_OPTIONS_CREATE_PCH -Wno-pch-messages -pch-create <PCH_FILE> -include <PCH_HEADER>)

      # COMPILE_WARNING_AS_ERROR
      set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "-Werror-all")
    endif()

    set(CMAKE_${lang}_LINK_MODE DRIVER)
  endmacro()
endif()
