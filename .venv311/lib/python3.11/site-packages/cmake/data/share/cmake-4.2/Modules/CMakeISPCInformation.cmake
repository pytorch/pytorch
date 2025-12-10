# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

if(UNIX)
  set(CMAKE_ISPC_OUTPUT_EXTENSION .o)
else()
  set(CMAKE_ISPC_OUTPUT_EXTENSION .obj)
endif()
set(CMAKE_INCLUDE_FLAG_ISPC "-I")

# Load compiler-specific information.
if(CMAKE_ISPC_COMPILER_ID)
  include(Compiler/${CMAKE_ISPC_COMPILER_ID}-ISPC OPTIONAL)
endif()

# load the system- and compiler specific files
if(CMAKE_ISPC_COMPILER_ID)
  # load a hardware specific file, mostly useful for embedded compilers
  if(CMAKE_SYSTEM_PROCESSOR)
    include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_ISPC_COMPILER_ID}-ISPC-${CMAKE_SYSTEM_PROCESSOR} OPTIONAL)
  endif()
  include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_ISPC_COMPILER_ID}-ISPC OPTIONAL)
endif()

# add the flags to the cache based
# on the initial values computed in the platform/*.cmake files
# use _INIT variables so that this only happens the first time
# and you can set these flags in the cmake cache
set(CMAKE_ISPC_FLAGS_INIT "$ENV{ISPCFLAGS} ${CMAKE_ISPC_FLAGS_INIT}")

cmake_initialize_per_config_variable(CMAKE_ISPC_FLAGS "Flags used by the ISPC compiler")

if(CMAKE_ISPC_STANDARD_LIBRARIES_INIT)
  set(CMAKE_ISPC_STANDARD_LIBRARIES "${CMAKE_ISPC_STANDARD_LIBRARIES_INIT}"
    CACHE STRING "Libraries linked by default with all ISPC applications.")
  mark_as_advanced(CMAKE_ISPC_STANDARD_LIBRARIES)
endif()

if(NOT CMAKE_ISPC_COMPILER_LAUNCHER AND DEFINED ENV{CMAKE_ISPC_COMPILER_LAUNCHER})
  set(CMAKE_ISPC_COMPILER_LAUNCHER "$ENV{CMAKE_ISPC_COMPILER_LAUNCHER}"
    CACHE STRING "Compiler launcher for ISPC.")
endif()

include(CMakeCommonLanguageInclude)

# now define the following rules:
# CMAKE_ISPC_COMPILE_OBJECT

# Create a static archive incrementally for large object file counts.
if(NOT DEFINED CMAKE_ISPC_ARCHIVE_CREATE)
  set(CMAKE_ISPC_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_ISPC_ARCHIVE_APPEND)
  set(CMAKE_ISPC_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_ISPC_ARCHIVE_FINISH)
  set(CMAKE_ISPC_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
endif()

if(NOT CMAKE_ISPC_COMPILE_OBJECT)
  set(CMAKE_ISPC_COMPILE_OBJECT
    "<CMAKE_ISPC_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> --emit-obj <SOURCE> -h <ISPC_HEADER>")
endif()

set(CMAKE_ISPC_USE_LINKER_INFORMATION FALSE)

set(CMAKE_ISPC_INFORMATION_LOADED 1)
