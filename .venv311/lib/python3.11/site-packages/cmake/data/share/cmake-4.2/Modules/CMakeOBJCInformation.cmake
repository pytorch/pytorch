# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This file sets the basic flags for the Objective-C language in CMake.
# It also loads the available platform file for the system-compiler
# if it exists.
# It also loads a system - compiler - processor (or target hardware)
# specific file, which is mainly useful for crosscompiling and embedded systems.

include(CMakeLanguageInformation)

# some compilers use different extensions (e.g. sdcc uses .rel)
# so set the extension here first so it can be overridden by the compiler specific file
set(CMAKE_OBJC_OUTPUT_EXTENSION .o)

if(NOT CMAKE_INCLUDE_FLAG_OBJC)
  set(CMAKE_INCLUDE_FLAG_OBJC ${CMAKE_INCLUDE_FLAG_C})
endif()

set(_INCLUDED_FILE 0)

# Load compiler-specific information.
if(CMAKE_OBJC_COMPILER_ID)
  include(Compiler/${CMAKE_OBJC_COMPILER_ID}-OBJC OPTIONAL)
endif()

set(CMAKE_BASE_NAME)
get_filename_component(CMAKE_BASE_NAME "${CMAKE_OBJC_COMPILER}" NAME_WE)
if(CMAKE_OBJC_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_BASE_NAME gcc)
endif()


# load a hardware specific file, mostly useful for embedded compilers
if(CMAKE_SYSTEM_PROCESSOR)
  if(CMAKE_OBJC_COMPILER_ID)
    include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_OBJC_COMPILER_ID}-OBJC-${CMAKE_SYSTEM_PROCESSOR} OPTIONAL RESULT_VARIABLE _INCLUDED_FILE)
  endif()
  if (NOT _INCLUDED_FILE)
    include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_BASE_NAME}-${CMAKE_SYSTEM_PROCESSOR} OPTIONAL)
  endif ()
endif()


# load the system- and compiler specific files
if(CMAKE_OBJC_COMPILER_ID)
  include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_OBJC_COMPILER_ID}-OBJC
    OPTIONAL RESULT_VARIABLE _INCLUDED_FILE)
endif()
if (NOT _INCLUDED_FILE)
  include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_BASE_NAME}
    OPTIONAL RESULT_VARIABLE _INCLUDED_FILE)
endif ()

# load any compiler-wrapper specific information
if (CMAKE_OBJC_COMPILER_WRAPPER)
  __cmake_include_compiler_wrapper(OBJC)
endif ()

# We specify the compiler information in the system file for some
# platforms, but this language may not have been enabled when the file
# was first included.  Include it again to get the language info.
# Remove this when all compiler info is removed from system files.
if (NOT _INCLUDED_FILE)
  include(Platform/${CMAKE_SYSTEM_NAME} OPTIONAL)
endif ()

if(CMAKE_OBJC_SIZEOF_DATA_PTR)
  foreach(f IN LISTS CMAKE_OBJC_ABI_FILES)
    include(${f})
  endforeach()
  unset(CMAKE_OBJC_ABI_FILES)
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

if(CMAKE_USER_MAKE_RULES_OVERRIDE_OBJC)
  # Save the full path of the file so try_compile can use it.
  include(${CMAKE_USER_MAKE_RULES_OVERRIDE_OBJC} RESULT_VARIABLE _override)
  set(CMAKE_USER_MAKE_RULES_OVERRIDE_OBJC "${_override}")
endif()

set(CMAKE_OBJC_FLAGS_INIT "$ENV{OBJCFLAGS} ${CMAKE_OBJC_FLAGS_INIT}")

cmake_initialize_per_config_variable(CMAKE_OBJC_FLAGS "Flags used by the Objective-C compiler")

if(CMAKE_OBJC_STANDARD_LIBRARIES_INIT)
  set(CMAKE_OBJC_STANDARD_LIBRARIES "${CMAKE_OBJC_STANDARD_LIBRARIES_INIT}"
    CACHE STRING "Libraries linked by default with all Objective-C applications.")
  mark_as_advanced(CMAKE_OBJC_STANDARD_LIBRARIES)
endif()

if(NOT CMAKE_OBJC_COMPILER_LAUNCHER AND DEFINED ENV{CMAKE_OBJC_COMPILER_LAUNCHER})
  set(CMAKE_OBJC_COMPILER_LAUNCHER "$ENV{CMAKE_OBJC_COMPILER_LAUNCHER}"
    CACHE STRING "Compiler launcher for OBJC.")
endif()

if(NOT CMAKE_OBJC_LINKER_LAUNCHER AND DEFINED ENV{CMAKE_OBJC_LINKER_LAUNCHER})
  set(CMAKE_OBJC_LINKER_LAUNCHER "$ENV{CMAKE_OBJC_LINKER_LAUNCHER}"
    CACHE STRING "Linker launcher for OBJC.")
endif()

include(CMakeCommonLanguageInclude)
_cmake_common_language_platform_flags(OBJC)

# now define the following rule variables

# CMAKE_OBJC_CREATE_SHARED_LIBRARY
# CMAKE_OBJC_CREATE_SHARED_MODULE
# CMAKE_OBJC_COMPILE_OBJECT
# CMAKE_OBJC_LINK_EXECUTABLE

# variables supplied by the generator at use time
# <TARGET>
# <TARGET_BASE> the target without the suffix
# <OBJECTS>
# <OBJECT>
# <LINK_LIBRARIES>
# <FLAGS>
# <LINK_FLAGS>

# Objective-C compiler information
# <CMAKE_OBJC_COMPILER>

# Static library tools
# <CMAKE_AR>
# <CMAKE_RANLIB>


# create an Objective-C shared library
if(NOT CMAKE_OBJC_CREATE_SHARED_LIBRARY)
  set(CMAKE_OBJC_CREATE_SHARED_LIBRARY
      "<CMAKE_OBJC_COMPILER> <CMAKE_SHARED_LIBRARY_OBJC_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
endif()

# create an Objective-C shared module just copy the shared library rule
if(NOT CMAKE_OBJC_CREATE_SHARED_MODULE)
  set(CMAKE_OBJC_CREATE_SHARED_MODULE ${CMAKE_OBJC_CREATE_SHARED_LIBRARY})
endif()

# Create an static archive incrementally for large object file counts.
# If CMAKE_OBJC_CREATE_STATIC_LIBRARY is set it will override these.
if(NOT DEFINED CMAKE_OBJC_ARCHIVE_CREATE)
  set(CMAKE_OBJC_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_OBJC_ARCHIVE_APPEND)
  set(CMAKE_OBJC_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_OBJC_ARCHIVE_FINISH)
  set(CMAKE_OBJC_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
endif()

# compile an Objective-C file into an object file
if(NOT CMAKE_OBJC_COMPILE_OBJECT)
  set(CMAKE_OBJC_COMPILE_OBJECT
    "<CMAKE_OBJC_COMPILER> <DEFINES> <INCLUDES> -x objective-c <FLAGS> -o <OBJECT> -c <SOURCE>")
endif()

if(NOT CMAKE_OBJC_LINK_EXECUTABLE)
  set(CMAKE_OBJC_LINK_EXECUTABLE
    "<CMAKE_OBJC_COMPILER> <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()

if(NOT CMAKE_EXECUTABLE_RUNTIME_OBJC_FLAG)
  set(CMAKE_EXECUTABLE_RUNTIME_OBJC_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_OBJC_FLAG})
endif()

if(NOT CMAKE_EXECUTABLE_RUNTIME_OBJC_FLAG_SEP)
  set(CMAKE_EXECUTABLE_RUNTIME_OBJC_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_OBJC_FLAG_SEP})
endif()

if(NOT CMAKE_EXECUTABLE_RPATH_LINK_OBJC_FLAG)
  set(CMAKE_EXECUTABLE_RPATH_LINK_OBJC_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_OBJC_FLAG})
endif()

set(CMAKE_OBJC_USE_LINKER_INFORMATION TRUE)

set(CMAKE_OBJC_INFORMATION_LOADED 1)
