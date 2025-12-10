# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# this file has flags that are shared across languages and sets
# cache values that can be initialized in the platform-compiler.cmake file
# it may be included by more than one language.

string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " $ENV{LDFLAGS}")
string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT " $ENV{LDFLAGS}")
string(APPEND CMAKE_MODULE_LINKER_FLAGS_INIT " $ENV{LDFLAGS}")

cmake_initialize_per_config_variable(CMAKE_EXE_LINKER_FLAGS    "Flags used by the linker")
cmake_initialize_per_config_variable(CMAKE_SHARED_LINKER_FLAGS "Flags used by the linker during the creation of shared libraries")
cmake_initialize_per_config_variable(CMAKE_MODULE_LINKER_FLAGS "Flags used by the linker during the creation of modules")
cmake_initialize_per_config_variable(CMAKE_STATIC_LINKER_FLAGS "Flags used by the archiver during the creation of static libraries")

# Alias the build tool variable for backward compatibility.
set(CMAKE_BUILD_TOOL ${CMAKE_MAKE_PROGRAM})

mark_as_advanced(
CMAKE_VERBOSE_MAKEFILE
)

# The Platform/* modules set a bunch of platform-specific flags expressed
# for the C toolchain.  Other languages call this to copy them as defaults.
macro(_cmake_common_language_platform_flags lang)
  if(NOT DEFINED CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS)
    set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS ${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS})
  endif()

  if(NOT DEFINED CMAKE_${lang}_COMPILE_OPTIONS_PIC)
    set(CMAKE_${lang}_COMPILE_OPTIONS_PIC ${CMAKE_C_COMPILE_OPTIONS_PIC})
  endif()

  if(NOT DEFINED CMAKE_${lang}_COMPILE_OPTIONS_PIE)
    set(CMAKE_${lang}_COMPILE_OPTIONS_PIE ${CMAKE_C_COMPILE_OPTIONS_PIE})
  endif()
  if(NOT DEFINED CMAKE_${lang}_LINK_OPTIONS_PIE)
    set(CMAKE_${lang}_LINK_OPTIONS_PIE ${CMAKE_C_LINK_OPTIONS_PIE})
  endif()
  if(NOT DEFINED CMAKE_${lang}_LINK_OPTIONS_NO_PIE)
    set(CMAKE_${lang}_LINK_OPTIONS_NO_PIE ${CMAKE_C_LINK_OPTIONS_NO_PIE})
  endif()

  if(NOT DEFINED CMAKE_${lang}_COMPILE_OPTIONS_DLL)
    set(CMAKE_${lang}_COMPILE_OPTIONS_DLL ${CMAKE_C_COMPILE_OPTIONS_DLL})
  endif()

  if(NOT DEFINED CMAKE_SHARED_LIBRARY_${lang}_FLAGS)
    set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS ${CMAKE_SHARED_LIBRARY_C_FLAGS})
  endif()

  if(NOT DEFINED CMAKE_SHARED_LIBRARY_LINK_${lang}_FLAGS)
    set(CMAKE_SHARED_LIBRARY_LINK_${lang}_FLAGS ${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS})
  endif()

  if(NOT DEFINED CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG)
    set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG})
  endif()

  if(NOT DEFINED CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG_SEP)
    set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG_SEP})
  endif()

  if(NOT DEFINED CMAKE_SHARED_LIBRARY_RPATH_LINK_${lang}_FLAG)
    set(CMAKE_SHARED_LIBRARY_RPATH_LINK_${lang}_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_C_FLAG})
  endif()

  if(NOT DEFINED CMAKE_EXE_EXPORTS_${lang}_FLAG)
    set(CMAKE_EXE_EXPORTS_${lang}_FLAG ${CMAKE_EXE_EXPORTS_C_FLAG})
  endif()

  if(NOT DEFINED CMAKE_SHARED_LIBRARY_SONAME_${lang}_FLAG)
    set(CMAKE_SHARED_LIBRARY_SONAME_${lang}_FLAG ${CMAKE_SHARED_LIBRARY_SONAME_C_FLAG})
  endif()

  if(NOT DEFINED CMAKE_EXECUTABLE_RUNTIME_${lang}_FLAG)
    set(CMAKE_EXECUTABLE_RUNTIME_${lang}_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG})
  endif()

  if(NOT DEFINED CMAKE_EXECUTABLE_RUNTIME_${lang}_FLAG_SEP)
    set(CMAKE_EXECUTABLE_RUNTIME_${lang}_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG_SEP})
  endif()

  if(NOT DEFINED CMAKE_EXECUTABLE_RPATH_LINK_${lang}_FLAG)
    set(CMAKE_EXECUTABLE_RPATH_LINK_${lang}_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_${lang}_FLAG})
  endif()

  if(NOT DEFINED CMAKE_SHARED_LIBRARY_LINK_${lang}_WITH_RUNTIME_PATH)
    set(CMAKE_SHARED_LIBRARY_LINK_${lang}_WITH_RUNTIME_PATH ${CMAKE_SHARED_LIBRARY_LINK_C_WITH_RUNTIME_PATH})
  endif()

  if(NOT DEFINED CMAKE_INCLUDE_FLAG_${lang})
    set(CMAKE_INCLUDE_FLAG_${lang} ${CMAKE_INCLUDE_FLAG_C})
  endif()

  # for most systems a module is the same as a shared library
  # so unless the variable CMAKE_MODULE_EXISTS is set just
  # copy the values from the LIBRARY variables
  if(NOT CMAKE_MODULE_EXISTS)
    set(CMAKE_SHARED_MODULE_${lang}_FLAGS "${CMAKE_SHARED_LIBRARY_${lang}_FLAGS}")
    set(CMAKE_SHARED_MODULE_CREATE_${lang}_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS}")
  endif()

  if(NOT DEFINED CMAKE_SHARED_MODULE_CREATE_${lang}_FLAGS)
    set(CMAKE_SHARED_MODULE_CREATE_${lang}_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS}")
  endif()
  if(NOT DEFINED CMAKE_SHARED_MODULE_${lang}_FLAGS)
    set(CMAKE_SHARED_MODULE_${lang}_FLAGS "${CMAKE_SHARED_LIBRARY_C_FLAGS}")
  endif()

  foreach(type IN ITEMS SHARED_LIBRARY SHARED_MODULE EXE)
    if(NOT DEFINED CMAKE_${type}_LINK_STATIC_${lang}_FLAGS)
      set(CMAKE_${type}_LINK_STATIC_${lang}_FLAGS
        ${CMAKE_${type}_LINK_STATIC_C_FLAGS})
    endif()
    if(NOT DEFINED CMAKE_${type}_LINK_DYNAMIC_${lang}_FLAGS)
      set(CMAKE_${type}_LINK_DYNAMIC_${lang}_FLAGS
        ${CMAKE_${type}_LINK_DYNAMIC_C_FLAGS})
    endif()
  endforeach()
endmacro()
