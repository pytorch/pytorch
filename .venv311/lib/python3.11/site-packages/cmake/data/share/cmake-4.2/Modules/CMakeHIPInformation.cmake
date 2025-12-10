# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

if(UNIX)
  set(CMAKE_HIP_OUTPUT_EXTENSION .o)
else()
  set(CMAKE_HIP_OUTPUT_EXTENSION .obj)
endif()
set(CMAKE_INCLUDE_FLAG_HIP "-I")

# Set implicit links early so compiler-specific modules can use them.
set(__IMPLICIT_LINKS)
foreach(dir ${CMAKE_HIP_HOST_IMPLICIT_LINK_DIRECTORIES})
  string(APPEND __IMPLICIT_LINKS " -L\"${dir}\"")
endforeach()
foreach(lib ${CMAKE_HIP_HOST_IMPLICIT_LINK_LIBRARIES})
  if(${lib} MATCHES "/")
    string(APPEND __IMPLICIT_LINKS " \"${lib}\"")
  else()
    string(APPEND __IMPLICIT_LINKS " -l${lib}")
  endif()
endforeach()

# Load compiler-specific information.
if(CMAKE_HIP_COMPILER_ID)
  include(Compiler/${CMAKE_HIP_COMPILER_ID}-HIP OPTIONAL)
endif()

# load the system- and compiler specific files
if(CMAKE_HIP_COMPILER_ID)
  # load a hardware specific file, mostly useful for embedded compilers
  if(CMAKE_SYSTEM_PROCESSOR)
    include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_HIP_COMPILER_ID}-HIP-${CMAKE_SYSTEM_PROCESSOR} OPTIONAL)
  endif()
  include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_HIP_COMPILER_ID}-HIP OPTIONAL)
endif()


# add the flags to the cache based
# on the initial values computed in the platform/*.cmake files
# use _INIT variables so that this only happens the first time
# and you can set these flags in the cmake cache
set(CMAKE_HIP_FLAGS_INIT "$ENV{HIPFLAGS} ${CMAKE_HIP_FLAGS_INIT}")

cmake_initialize_per_config_variable(CMAKE_HIP_FLAGS "Flags used by the HIP compiler")

if(CMAKE_HIP_STANDARD_LIBRARIES_INIT)
  set(CMAKE_HIP_STANDARD_LIBRARIES "${CMAKE_HIP_STANDARD_LIBRARIES_INIT}"
    CACHE STRING "Libraries linked by default with all HIP applications.")
  mark_as_advanced(CMAKE_HIP_STANDARD_LIBRARIES)
endif()

if(NOT CMAKE_HIP_COMPILER_LAUNCHER AND DEFINED ENV{CMAKE_HIP_COMPILER_LAUNCHER})
  set(CMAKE_HIP_COMPILER_LAUNCHER "$ENV{CMAKE_HIP_COMPILER_LAUNCHER}"
    CACHE STRING "Compiler launcher for HIP.")
endif()

if(NOT CMAKE_HIP_LINKER_LAUNCHER AND DEFINED ENV{CMAKE_HIP_LINKER_LAUNCHER})
  set(CMAKE_HIP_LINKER_LAUNCHER "$ENV{CMAKE_HIP_LINKER_LAUNCHER}"
          CACHE STRING "Linker launcher for HIP.")
endif()

include(CMakeCommonLanguageInclude)
_cmake_common_language_platform_flags(HIP)

# now define the following rules:
# CMAKE_HIP_CREATE_SHARED_LIBRARY
# CMAKE_HIP_CREATE_SHARED_MODULE
# CMAKE_HIP_COMPILE_OBJECT
# CMAKE_HIP_LINK_EXECUTABLE

# create a shared library
if(NOT CMAKE_HIP_CREATE_SHARED_LIBRARY)
  set(CMAKE_HIP_CREATE_SHARED_LIBRARY
      "<CMAKE_HIP_COMPILER> <CMAKE_SHARED_LIBRARY_HIP_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
endif()

# create a shared module copy the shared library rule by default
if(NOT CMAKE_HIP_CREATE_SHARED_MODULE)
  set(CMAKE_HIP_CREATE_SHARED_MODULE ${CMAKE_HIP_CREATE_SHARED_LIBRARY})
endif()

# Create a static archive incrementally for large object file counts.
if(NOT DEFINED CMAKE_HIP_ARCHIVE_CREATE)
  set(CMAKE_HIP_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_HIP_ARCHIVE_APPEND)
  set(CMAKE_HIP_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_HIP_ARCHIVE_FINISH)
  set(CMAKE_HIP_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
endif()

# compile a HIP file into an object file
if(NOT CMAKE_HIP_COMPILE_OBJECT)
  set(CMAKE_HIP_COMPILE_OBJECT
    "<CMAKE_HIP_COMPILER> ${_CMAKE_HIP_EXTRA_FLAGS} <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> ${_CMAKE_COMPILE_AS_HIP_FLAG} -c <SOURCE>")
endif()

# compile a cu file into an executable
if(NOT CMAKE_HIP_LINK_EXECUTABLE)
  set(CMAKE_HIP_LINK_EXECUTABLE
    "<CMAKE_HIP_COMPILER> <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()

set(CMAKE_HIP_USE_LINKER_INFORMATION TRUE)

set(CMAKE_HIP_INFORMATION_LOADED 1)

# Load the file and find the relevant HIP runtime.
if(NOT DEFINED _CMAKE_HIP_DEVICE_RUNTIME_TARGET)
  set(hip-lang_DIR "${CMAKE_HIP_COMPILER_ROCM_LIB}/cmake/hip-lang")
  find_package(hip-lang CONFIG QUIET NO_DEFAULT_PATH REQUIRED)
endif()
if(DEFINED _CMAKE_HIP_DEVICE_RUNTIME_TARGET)
  list(APPEND CMAKE_HIP_RUNTIME_LIBRARIES_STATIC ${_CMAKE_HIP_DEVICE_RUNTIME_TARGET})
  list(APPEND CMAKE_HIP_RUNTIME_LIBRARIES_SHARED ${_CMAKE_HIP_DEVICE_RUNTIME_TARGET})
endif()
