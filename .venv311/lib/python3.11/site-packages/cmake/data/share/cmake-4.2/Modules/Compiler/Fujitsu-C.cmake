# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.
include(Compiler/Fujitsu)
__compiler_fujitsu(C)

if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 4)
  set(CMAKE_C90_STANDARD_COMPILE_OPTION  -std=c89)
  set(CMAKE_C90_EXTENSION_COMPILE_OPTION -std=gnu89)
  set(CMAKE_C90_STANDARD__HAS_FULL_SUPPORT ON)

  set(CMAKE_C99_STANDARD_COMPILE_OPTION  -std=c99)
  set(CMAKE_C99_EXTENSION_COMPILE_OPTION -std=gnu99)
  set(CMAKE_C99_STANDARD__HAS_FULL_SUPPORT ON)

  set(CMAKE_C11_STANDARD_COMPILE_OPTION  -std=c11)
  set(CMAKE_C11_EXTENSION_COMPILE_OPTION -std=gnu11)
  set(CMAKE_C11_STANDARD__HAS_FULL_SUPPORT ON)

  set(CMAKE_C_STANDARD_LATEST 11)
endif()

__compiler_check_default_language_standard(C 4 11)
