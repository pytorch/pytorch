# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This file is included by CMakeFindEclipseCDT4.cmake and CMakeFindCodeBlocks.cmake

# The Eclipse and the CodeBlocks generators need to know the standard include path
# so that they can find the headers at runtime and parsing etc. works better
# This is done here by actually running gcc with the options so it prints its
# system include directories, which are parsed then and stored in the cache.
macro(_DETERMINE_GCC_SYSTEM_INCLUDE_DIRS _lang _resultIncludeDirs _resultDefines)
  set(${_resultIncludeDirs})
  set(_gccOutput)
  file(WRITE "${CMAKE_BINARY_DIR}/CMakeFiles/dummy" "\n" )

  if (${_lang} STREQUAL "c++")
    set(_compilerExecutable "${CMAKE_CXX_COMPILER}")
    set(_arg1 "${CMAKE_CXX_COMPILER_ARG1}")

    if (CMAKE_CXX_FLAGS MATCHES "(-stdlib=[^ ]+)")
      set(_stdlib "${CMAKE_MATCH_1}")
    endif ()
    if (CMAKE_CXX_FLAGS MATCHES "(-std=[^ ]+)")
      set(_stdver "${CMAKE_MATCH_1}")
    endif ()
  else ()
    set(_compilerExecutable "${CMAKE_C_COMPILER}")
    set(_arg1 "${CMAKE_C_COMPILER_ARG1}")
  endif ()
  separate_arguments(_arg1 NATIVE_COMMAND "${_arg1}")
  execute_process(COMMAND ${_compilerExecutable} ${_arg1} ${_stdver} ${_stdlib} -v -E -x ${_lang} -dD dummy
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/CMakeFiles
                  ERROR_VARIABLE _gccOutput
                  OUTPUT_VARIABLE _gccStdout )
  file(REMOVE "${CMAKE_BINARY_DIR}/CMakeFiles/dummy")

  # First find the system include dirs:
  if( "${_gccOutput}" MATCHES "> search starts here[^\n]+\n *(.+ *\n) *End of (search) list" )

    # split the output into lines and then remove leading and trailing spaces from each of them:
    string(REGEX MATCHALL "[^\n]+\n" _includeLines "${CMAKE_MATCH_1}")
    foreach(nextLine ${_includeLines})
      # on OSX, gcc says things like this:  "/System/Library/Frameworks (framework directory)", strip the last part
      string(REGEX REPLACE "\\(framework directory\\)" "" nextLineNoFramework "${nextLine}")
      # strip spaces at the beginning and the end
      string(STRIP "${nextLineNoFramework}" _includePath)
      list(APPEND ${_resultIncludeDirs} "${_includePath}")
    endforeach()

  endif()


  # now find the builtin macros:
  string(REGEX MATCHALL "#define[^\n]+\n" _defineLines "${_gccStdout}")
# A few example lines which the regexp below has to match properly:
#  #define   MAX(a,b) ((a) > (b) ? (a) : (b))
#  #define __fastcall __attribute__((__fastcall__))
#  #define   FOO (23)
#  #define __UINTMAX_TYPE__ long long unsigned int
#  #define __UINTMAX_TYPE__ long long unsigned int
#  #define __i386__  1

  foreach(nextLine ${_defineLines})
    string(REGEX MATCH "^#define +([A-Za-z_][A-Za-z0-9_]*)(\\([^\\)]+\\))? +(.+) *$" _dummy "${nextLine}")
    set(_name "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
    string(STRIP "${CMAKE_MATCH_3}" _value)
    #message(STATUS "m1: -${CMAKE_MATCH_1}- m2: -${CMAKE_MATCH_2}- m3: -${CMAKE_MATCH_3}-")

    list(APPEND ${_resultDefines} "${_name}")
    if ("${_value}" STREQUAL "")
      list(APPEND ${_resultDefines} " ")
    else()
      list(APPEND ${_resultDefines} "${_value}")
    endif()
  endforeach()

endmacro()

# Save the current LC_ALL, LC_MESSAGES, and LANG environment variables and set them
# to "C" that way GCC's "search starts here" text is in English and we can grok it.
set(_orig_lc_all      $ENV{LC_ALL})
set(_orig_lc_messages $ENV{LC_MESSAGES})
set(_orig_lang        $ENV{LANG})

set(ENV{LC_ALL}      C)
set(ENV{LC_MESSAGES} C)
set(ENV{LANG}        C)

# Now check for C, works for gcc and Intel compiler at least
if (NOT CMAKE_EXTRA_GENERATOR_C_SYSTEM_INCLUDE_DIRS)
  if (CMAKE_C_COMPILER_ID MATCHES GNU  OR  CMAKE_C_COMPILER_ID MATCHES "LCC"  OR  CMAKE_C_COMPILER_ID MATCHES "Intel"  OR  CMAKE_C_COMPILER_ID MATCHES Clang)
    _DETERMINE_GCC_SYSTEM_INCLUDE_DIRS(c _dirs _defines)
    set(CMAKE_EXTRA_GENERATOR_C_SYSTEM_INCLUDE_DIRS "${_dirs}" CACHE INTERNAL "C compiler system include directories")
    set(CMAKE_EXTRA_GENERATOR_C_SYSTEM_DEFINED_MACROS "${_defines}" CACHE INTERNAL "C compiler system defined macros")
  elseif ("${CMAKE_C_COMPILER_ID}" MATCHES MSVC)
    set(CMAKE_EXTRA_GENERATOR_C_SYSTEM_INCLUDE_DIRS "$ENV{INCLUDE}" CACHE INTERNAL "C compiler system include directories")
  endif ()
endif ()

# And now the same for C++
if (NOT CMAKE_EXTRA_GENERATOR_CXX_SYSTEM_INCLUDE_DIRS)
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES GNU  OR  "${CMAKE_CXX_COMPILER_ID}" MATCHES "LCC"  OR  "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel"  OR  "${CMAKE_CXX_COMPILER_ID}" MATCHES Clang)
    _DETERMINE_GCC_SYSTEM_INCLUDE_DIRS(c++ _dirs _defines)
    set(CMAKE_EXTRA_GENERATOR_CXX_SYSTEM_INCLUDE_DIRS "${_dirs}" CACHE INTERNAL "CXX compiler system include directories")
    set(CMAKE_EXTRA_GENERATOR_CXX_SYSTEM_DEFINED_MACROS "${_defines}" CACHE INTERNAL "CXX compiler system defined macros")
  elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES MSVC)
    set(CMAKE_EXTRA_GENERATOR_CXX_SYSTEM_INCLUDE_DIRS "$ENV{INCLUDE}" CACHE INTERNAL "CXX compiler system include directories")
  endif ()
endif ()

# Restore original LC_ALL, LC_MESSAGES, and LANG
set(ENV{LC_ALL}      ${_orig_lc_all})
set(ENV{LC_MESSAGES} ${_orig_lc_messages})
set(ENV{LANG}        ${_orig_lang})
