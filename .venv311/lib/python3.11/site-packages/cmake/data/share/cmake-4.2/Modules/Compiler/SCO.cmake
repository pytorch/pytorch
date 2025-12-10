# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__COMPILER_SCO)
  return()
endif()
set(__COMPILER_SCO 1)

macro(__compiler_sco lang)
  # Feature flags.
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC -Kpic)
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIE -Kpie)
  set(CMAKE_${lang}_COMPILE_OPTIONS_DLL -belf)
  set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS "-Kpic -belf")
  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-belf -Wl,-Bexport")

  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-Wl,")
  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG_SEP ",")

  set(CMAKE_${lang}_LINK_MODE DRIVER)
endmacro()
