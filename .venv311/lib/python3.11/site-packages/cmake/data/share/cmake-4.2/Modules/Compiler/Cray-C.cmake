# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include(Compiler/Cray)
__compiler_cray(C)

string(APPEND CMAKE_C_FLAGS_MINSIZEREL_INIT " -DNDEBUG")
string(APPEND CMAKE_C_FLAGS_RELEASE_INIT " -DNDEBUG")

if (CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 8.1)
  set(CMAKE_C90_STANDARD_COMPILE_OPTION  -h noc99,conform)
  set(CMAKE_C90_EXTENSION_COMPILE_OPTION -h noc99,gnu)
  set(CMAKE_C90_STANDARD__HAS_FULL_SUPPORT ON)
  set(CMAKE_C99_STANDARD_COMPILE_OPTION  -h c99,conform)
  set(CMAKE_C99_EXTENSION_COMPILE_OPTION -h c99,gnu)
  set(CMAKE_C99_STANDARD__HAS_FULL_SUPPORT ON)
  set(CMAKE_C_STANDARD_LATEST 99)
  if (CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 8.5)
    set(CMAKE_C11_STANDARD_COMPILE_OPTION  -h std=c11,conform)
    set(CMAKE_C11_EXTENSION_COMPILE_OPTION -h std=c11,gnu)
    set(CMAKE_C11_STANDARD__HAS_FULL_SUPPORT ON)
    set(CMAKE_C_STANDARD_LATEST 11)
  endif ()
endif ()

__compiler_check_default_language_standard(C 8.1 99)
