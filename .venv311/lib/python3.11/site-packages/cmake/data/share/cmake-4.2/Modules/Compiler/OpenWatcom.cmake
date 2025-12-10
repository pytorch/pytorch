# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This module is shared by multiple languages; use include blocker.
include_guard()

set(CMAKE_LIBRARY_PATH_FLAG "libpath ")
set(CMAKE_LINK_LIBRARY_FLAG "library ")
set(CMAKE_LINK_LIBRARY_FILE_FLAG "library ")
set(CMAKE_LINK_OBJECT_FILE_FLAG "file ")

if(CMAKE_VERBOSE_MAKEFILE)
  set(CMAKE_WCL_QUIET)
  set(CMAKE_WLINK_QUIET)
  set(CMAKE_LIB_QUIET)
else()
  set(CMAKE_WCL_QUIET "-zq")
  set(CMAKE_WLINK_QUIET "option quiet")
  set(CMAKE_LIB_QUIET "-q")
endif()

foreach(type CREATE_SHARED_LIBRARY CREATE_SHARED_MODULE LINK_EXECUTABLE)
  set(CMAKE_C_${type}_USE_WATCOM_QUOTE 1)
  set(CMAKE_CXX_${type}_USE_WATCOM_QUOTE 1)
endforeach()

foreach(type SHARED MODULE EXE)
  # linker map file creation directives
  string(APPEND CMAKE_${type}_LINKER_FLAGS_INIT " opt map")
  # linker debug directives
  string(APPEND CMAKE_${type}_LINKER_FLAGS_DEBUG_INIT " debug all")
  string(APPEND CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO_INIT " debug all")
endforeach()

foreach(lang C CXX)
  # warning level
  string(APPEND CMAKE_${lang}_FLAGS_INIT " -w3")
  # debug options
  string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -d2")
  string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -s -os -d0 -dNDEBUG")
  string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " -s -ot -d0 -dNDEBUG")
  string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " -s -ot -d1 -dNDEBUG")

  set(CMAKE_${lang}_LINK_MODE LINKER)
endforeach()

# C create import library
set(CMAKE_C_CREATE_IMPORT_LIBRARY
  "<CMAKE_AR> -c -q -n -b <TARGET_IMPLIB> +<TARGET_QUOTED>")
# C++ create import library
set(CMAKE_CXX_CREATE_IMPORT_LIBRARY ${CMAKE_C_CREATE_IMPORT_LIBRARY})

# C link a object files into an executable file
set(CMAKE_C_LINK_EXECUTABLE
  "<CMAKE_LINKER> ${CMAKE_WLINK_QUIET} name <TARGET> <LINK_FLAGS> file {<OBJECTS>} <LINK_LIBRARIES>")
# C++ link a object files into an executable file
set(CMAKE_CXX_LINK_EXECUTABLE ${CMAKE_C_LINK_EXECUTABLE})

# C compile a file into an object file
set(CMAKE_C_COMPILE_OBJECT
  "<CMAKE_C_COMPILER> ${CMAKE_WCL_QUIET} -d+ <DEFINES> <INCLUDES> <FLAGS> -fo<OBJECT> -c -cc <SOURCE>")
# C++ compile a file into an object file
set(CMAKE_CXX_COMPILE_OBJECT
  "<CMAKE_CXX_COMPILER> ${CMAKE_WCL_QUIET} -d+ <DEFINES> <INCLUDES> <FLAGS> -fo<OBJECT> -c -cc++ <SOURCE>")

# C preprocess a source file
set(CMAKE_C_CREATE_PREPROCESSED_SOURCE
  "<CMAKE_C_COMPILER> ${CMAKE_WCL_QUIET} -d+ <DEFINES> <INCLUDES> <FLAGS> -fo<PREPROCESSED_SOURCE> -pl -cc <SOURCE>")
# C++ preprocess a source file
set(CMAKE_CXX_CREATE_PREPROCESSED_SOURCE
  "<CMAKE_CXX_COMPILER> ${CMAKE_WCL_QUIET} -d+ <DEFINES> <INCLUDES> <FLAGS> -fo<PREPROCESSED_SOURCE> -pl -cc++ <SOURCE>")

# C create a shared library
set(CMAKE_C_CREATE_SHARED_LIBRARY
  "<CMAKE_LINKER> ${CMAKE_WLINK_QUIET} name <TARGET> <LINK_FLAGS> option implib=<TARGET_IMPLIB> file {<OBJECTS>} <LINK_LIBRARIES>")
# C++ create a shared library
set(CMAKE_CXX_CREATE_SHARED_LIBRARY ${CMAKE_C_CREATE_SHARED_LIBRARY})
set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "")
set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "")

# C create a shared module
set(CMAKE_C_CREATE_SHARED_MODULE
  "<CMAKE_LINKER> ${CMAKE_WLINK_QUIET} name <TARGET> <LINK_FLAGS> file {<OBJECTS>} <LINK_LIBRARIES>")
# C++ create a shared module
set(CMAKE_CXX_CREATE_SHARED_MODULE ${CMAKE_C_CREATE_SHARED_MODULE})
set(CMAKE_SHARED_MODULE_CREATE_C_FLAGS "")
set(CMAKE_SHARED_MODULE_CREATE_CXX_FLAGS "")

# C create a static library
set(CMAKE_C_CREATE_STATIC_LIBRARY
  "<CMAKE_AR> ${CMAKE_LIB_QUIET} -c -n -b <TARGET_QUOTED> <LINK_FLAGS> <OBJECTS> ")
# C++ create a static library
set(CMAKE_CXX_CREATE_STATIC_LIBRARY ${CMAKE_C_CREATE_STATIC_LIBRARY})


# old CMake internally used OpenWatcom version macros
# for backward compatibility
if(NOT _CMAKE_WATCOM_VERSION)
  set(_CMAKE_WATCOM_VERSION 1)
  if(CMAKE_C_COMPILER_VERSION)
    set(_compiler_version ${CMAKE_C_COMPILER_VERSION})
    set(_compiler_id ${CMAKE_C_COMPILER_ID})
  else()
    set(_compiler_version ${CMAKE_CXX_COMPILER_VERSION})
    set(_compiler_id ${CMAKE_CXX_COMPILER_ID})
  endif()
  set(WATCOM16)
  set(WATCOM17)
  set(WATCOM18)
  set(WATCOM19)
  if("${_compiler_id}" STREQUAL "OpenWatcom")
    if("${_compiler_version}" VERSION_LESS 1.7)
      set(WATCOM16 1)
    endif()
    if("${_compiler_version}" VERSION_EQUAL 1.7)
      set(WATCOM17 1)
    endif()
    if("${_compiler_version}" VERSION_EQUAL 1.8)
      set(WATCOM18 1)
    endif()
    if("${_compiler_version}" VERSION_EQUAL 1.9)
      set(WATCOM19 1)
    endif()
  endif()
endif()
