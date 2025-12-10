# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This file sets the basic flags for the C++ language in CMake.
# It also loads the available platform file for the system-compiler
# if it exists.
# It also loads a system - compiler - processor (or target hardware)
# specific file, which is mainly useful for crosscompiling and embedded systems.

include(CMakeLanguageInformation)

# some compilers use different extensions (e.g. sdcc uses .rel)
# so set the extension here first so it can be overridden by the compiler specific file
if(UNIX)
  set(CMAKE_CXX_OUTPUT_EXTENSION .o)
else()
  set(CMAKE_CXX_OUTPUT_EXTENSION .obj)
endif()

set(_INCLUDED_FILE 0)

# Load compiler-specific information.
if(CMAKE_CXX_COMPILER_ID)
  include(Compiler/${CMAKE_CXX_COMPILER_ID}-CXX OPTIONAL)
endif()

set(CMAKE_BASE_NAME)
get_filename_component(CMAKE_BASE_NAME "${CMAKE_CXX_COMPILER}" NAME_WE)
# since the gnu compiler has several names force g++
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_BASE_NAME g++)
endif()


# load a hardware specific file, mostly useful for embedded compilers
if(CMAKE_SYSTEM_PROCESSOR)
  if(CMAKE_CXX_COMPILER_ID)
    include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_CXX_COMPILER_ID}-CXX-${CMAKE_SYSTEM_PROCESSOR} OPTIONAL RESULT_VARIABLE _INCLUDED_FILE)
  endif()
  if (NOT _INCLUDED_FILE)
    include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_BASE_NAME}-${CMAKE_SYSTEM_PROCESSOR} OPTIONAL)
  endif ()
endif()

# load the system- and compiler specific files
if(CMAKE_CXX_COMPILER_ID)
  include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_CXX_COMPILER_ID}-CXX OPTIONAL RESULT_VARIABLE _INCLUDED_FILE)
endif()
if (NOT _INCLUDED_FILE)
  include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_BASE_NAME} OPTIONAL
          RESULT_VARIABLE _INCLUDED_FILE)
endif ()

# load any compiler-wrapper specific information
if (CMAKE_CXX_COMPILER_WRAPPER)
  __cmake_include_compiler_wrapper(CXX)
endif ()

# We specify the compiler information in the system file for some
# platforms, but this language may not have been enabled when the file
# was first included.  Include it again to get the language info.
# Remove this when all compiler info is removed from system files.
if (NOT _INCLUDED_FILE)
  include(Platform/${CMAKE_SYSTEM_NAME} OPTIONAL)
endif ()

if(CMAKE_CXX_SIZEOF_DATA_PTR)
  foreach(f IN LISTS CMAKE_CXX_ABI_FILES)
    include(${f})
  endforeach()
  unset(CMAKE_CXX_ABI_FILES)
endif()

# This should be included before the _INIT variables are
# used to initialize the cache.  Since the rule variables
# have if blocks on them, users can still define them here.
# But, it should still be after the platform file so changes can
# be made to those values.

if(CMAKE_USER_MAKE_RULES_OVERRIDE)
  # Save the full path of the file so try_compile can use it.
  include(${CMAKE_USER_MAKE_RULES_OVERRIDE} RESULT_VARIABLE _override)
  set(CMAKE_USER_MAKE_RULES_OVERRIDE "${_override}")
endif()

if(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX)
  # Save the full path of the file so try_compile can use it.
  include(${CMAKE_USER_MAKE_RULES_OVERRIDE_CXX} RESULT_VARIABLE _override)
  set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX "${_override}")
endif()


# add the flags to the cache based
# on the initial values computed in the platform/*.cmake files
# use _INIT variables so that this only happens the first time
# and you can set these flags in the cmake cache
set(CMAKE_CXX_FLAGS_INIT "$ENV{CXXFLAGS} ${CMAKE_CXX_FLAGS_INIT}")

cmake_initialize_per_config_variable(CMAKE_CXX_FLAGS "Flags used by the CXX compiler")

if(CMAKE_CXX_STANDARD_LIBRARIES_INIT)
  set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES_INIT}"
    CACHE STRING "Libraries linked by default with all C++ applications.")
  mark_as_advanced(CMAKE_CXX_STANDARD_LIBRARIES)
endif()

if(NOT CMAKE_CXX_COMPILER_LAUNCHER AND DEFINED ENV{CMAKE_CXX_COMPILER_LAUNCHER})
  set(CMAKE_CXX_COMPILER_LAUNCHER "$ENV{CMAKE_CXX_COMPILER_LAUNCHER}"
    CACHE STRING "Compiler launcher for CXX.")
endif()

if(NOT CMAKE_CXX_LINKER_LAUNCHER AND DEFINED ENV{CMAKE_CXX_LINKER_LAUNCHER})
  set(CMAKE_CXX_LINKER_LAUNCHER "$ENV{CMAKE_CXX_LINKER_LAUNCHER}"
    CACHE STRING "Linker launcher for CXX.")
endif()

include(CMakeCommonLanguageInclude)
_cmake_common_language_platform_flags(CXX)

# now define the following rules:
# CMAKE_CXX_CREATE_SHARED_LIBRARY
# CMAKE_CXX_CREATE_SHARED_MODULE
# CMAKE_CXX_COMPILE_OBJECT
# CMAKE_CXX_LINK_EXECUTABLE

# variables supplied by the generator at use time
# <TARGET>
# <TARGET_BASE> the target without the suffix
# <OBJECTS>
# <OBJECT>
# <LINK_LIBRARIES>
# <FLAGS>
# <LINK_FLAGS>

# CXX compiler information
# <CMAKE_CXX_COMPILER>

# Static library tools
# <CMAKE_AR>
# <CMAKE_RANLIB>


# create a shared C++ library
if(NOT CMAKE_CXX_CREATE_SHARED_LIBRARY)
  set(CMAKE_CXX_CREATE_SHARED_LIBRARY
      "<CMAKE_CXX_COMPILER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
endif()

# create a c++ shared module copy the shared library rule by default
if(NOT CMAKE_CXX_CREATE_SHARED_MODULE)
  set(CMAKE_CXX_CREATE_SHARED_MODULE ${CMAKE_CXX_CREATE_SHARED_LIBRARY})
endif()


# Create a static archive incrementally for large object file counts.
# If CMAKE_CXX_CREATE_STATIC_LIBRARY is set it will override these.
if(NOT DEFINED CMAKE_CXX_ARCHIVE_CREATE)
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_CXX_ARCHIVE_APPEND)
  set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_CXX_ARCHIVE_FINISH)
  set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
endif()

# compile a C++ file into an object file
if(NOT CMAKE_CXX_COMPILE_OBJECT)
  set(CMAKE_CXX_COMPILE_OBJECT
    "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")
endif()

if(NOT CMAKE_CXX_LINK_EXECUTABLE)
  set(CMAKE_CXX_LINK_EXECUTABLE
    "<CMAKE_CXX_COMPILER> <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()

mark_as_advanced(
CMAKE_VERBOSE_MAKEFILE
)

set(CMAKE_CXX_USE_LINKER_INFORMATION TRUE)

set(CMAKE_CXX_INFORMATION_LOADED 1)
