# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

if(UNIX)
  set(CMAKE_CUDA_OUTPUT_EXTENSION .o)
else()
  set(CMAKE_CUDA_OUTPUT_EXTENSION .obj)
endif()
set(CMAKE_INCLUDE_FLAG_CUDA "-I")

# Set implicit links early so compiler-specific modules can use them.
set(__IMPLICIT_LINKS)
foreach(dir ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES})
  string(APPEND __IMPLICIT_LINKS " -L\"${dir}\"")
endforeach()
foreach(lib ${CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES})
  if(${lib} MATCHES "/")
    string(APPEND __IMPLICIT_LINKS " \"${lib}\"")
  else()
    string(APPEND __IMPLICIT_LINKS " -l${lib}")
  endif()
endforeach()

# Load compiler-specific information.
if(CMAKE_CUDA_COMPILER_ID)
  include(Compiler/${CMAKE_CUDA_COMPILER_ID}-CUDA OPTIONAL)
endif()

# load the system- and compiler specific files
if(CMAKE_CUDA_COMPILER_ID)
  # load a hardware specific file, mostly useful for embedded compilers
  if(CMAKE_SYSTEM_PROCESSOR)
    include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_CUDA_COMPILER_ID}-CUDA-${CMAKE_SYSTEM_PROCESSOR} OPTIONAL)
  endif()
  include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_CUDA_COMPILER_ID}-CUDA OPTIONAL)
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

if(CMAKE_USER_MAKE_RULES_OVERRIDE_CUDA)
  # Save the full path of the file so try_compile can use it.
  include(${CMAKE_USER_MAKE_RULES_OVERRIDE_CUDA} RESULT_VARIABLE _override)
  set(CMAKE_USER_MAKE_RULES_OVERRIDE_CUDA "${_override}")
endif()



# add the flags to the cache based
# on the initial values computed in the platform/*.cmake files
# use _INIT variables so that this only happens the first time
# and you can set these flags in the cmake cache
set(CMAKE_CUDA_FLAGS_INIT "$ENV{CUDAFLAGS} ${CMAKE_CUDA_FLAGS_INIT}")

cmake_initialize_per_config_variable(CMAKE_CUDA_FLAGS "Flags used by the CUDA compiler")

if(CMAKE_CUDA_STANDARD_LIBRARIES_INIT)
  set(CMAKE_CUDA_STANDARD_LIBRARIES "${CMAKE_CUDA_STANDARD_LIBRARIES_INIT}"
    CACHE STRING "Libraries linked by default with all CUDA applications.")
  mark_as_advanced(CMAKE_CUDA_STANDARD_LIBRARIES)
endif()

if(NOT CMAKE_CUDA_COMPILER_LAUNCHER AND DEFINED ENV{CMAKE_CUDA_COMPILER_LAUNCHER})
  set(CMAKE_CUDA_COMPILER_LAUNCHER "$ENV{CMAKE_CUDA_COMPILER_LAUNCHER}"
    CACHE STRING "Compiler launcher for CUDA.")
endif()

if(NOT CMAKE_CUDA_LINKER_LAUNCHER AND DEFINED ENV{CMAKE_CUDA_LINKER_LAUNCHER})
  set(CMAKE_CUDA_LINKER_LAUNCHER "$ENV{CMAKE_CUDA_LINKER_LAUNCHER}"
          CACHE STRING "Linker launcher for CUDA.")
endif()

include(CMakeCommonLanguageInclude)
_cmake_common_language_platform_flags(CUDA)

# now define the following rules:
# CMAKE_CUDA_CREATE_SHARED_LIBRARY
# CMAKE_CUDA_CREATE_SHARED_MODULE
# CMAKE_CUDA_COMPILE_WHOLE_COMPILATION
# CMAKE_CUDA_COMPILE_SEPARABLE_COMPILATION
# CMAKE_CUDA_LINK_EXECUTABLE

# create a shared library
if(NOT CMAKE_CUDA_CREATE_SHARED_LIBRARY)
  set(CMAKE_CUDA_CREATE_SHARED_LIBRARY
      "<CMAKE_CUDA_HOST_LINK_LAUNCHER> <CMAKE_SHARED_LIBRARY_CUDA_FLAGS> <LINK_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>${__IMPLICIT_LINKS}")
endif()

# create a shared module copy the shared library rule by default
if(NOT CMAKE_CUDA_CREATE_SHARED_MODULE)
  set(CMAKE_CUDA_CREATE_SHARED_MODULE ${CMAKE_CUDA_CREATE_SHARED_LIBRARY})
endif()

# Create a static archive incrementally for large object file counts.
if(NOT DEFINED CMAKE_CUDA_ARCHIVE_CREATE)
  set(CMAKE_CUDA_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHIVE_APPEND)
  set(CMAKE_CUDA_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHIVE_FINISH)
  set(CMAKE_CUDA_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
endif()

if(NOT CMAKE_CUDA_COMPILE_OBJECT)
  set(CMAKE_CUDA_COMPILE_OBJECT
    "<CMAKE_CUDA_COMPILER> ${_CMAKE_CUDA_EXTRA_FLAGS} <DEFINES> <INCLUDES> <FLAGS> ${_CMAKE_COMPILE_AS_CUDA_FLAG} <CUDA_COMPILE_MODE> <SOURCE> -o <OBJECT>")
endif()

# compile a cu file into an executable
if(NOT CMAKE_CUDA_LINK_EXECUTABLE)
  set(CMAKE_CUDA_LINK_EXECUTABLE
    "<CMAKE_CUDA_HOST_LINK_LAUNCHER> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>${__IMPLICIT_LINKS}")
endif()

# Add implicit host link directories that contain device libraries
# to the device link line.
set(__IMPLICIT_DLINK_DIRS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
if(__IMPLICIT_DLINK_DIRS)
  list(REMOVE_ITEM __IMPLICIT_DLINK_DIRS ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES})
endif()
set(__IMPLICIT_DLINK_FLAGS)
foreach(dir ${__IMPLICIT_DLINK_DIRS})
  if(EXISTS "${dir}/libcurand_static.a")
    string(APPEND __IMPLICIT_DLINK_FLAGS " -L\"${dir}\"")
  endif()
endforeach()
unset(__IMPLICIT_DLINK_DIRS)


#These are used when linking relocatable (dc) cuda code
if(NOT CMAKE_CUDA_DEVICE_LINK_LIBRARY)
  set(CMAKE_CUDA_DEVICE_LINK_LIBRARY
    "<CMAKE_CUDA_COMPILER> ${_CMAKE_CUDA_EXTRA_FLAGS} <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> ${CMAKE_CUDA_COMPILE_OPTIONS_PIC} ${_CMAKE_CUDA_EXTRA_DEVICE_LINK_FLAGS} -shared -dlink <OBJECTS> -o <TARGET> <LINK_LIBRARIES>${__IMPLICIT_DLINK_FLAGS}")
endif()
if(NOT CMAKE_CUDA_DEVICE_LINK_EXECUTABLE)
  set(CMAKE_CUDA_DEVICE_LINK_EXECUTABLE
    "<CMAKE_CUDA_COMPILER> ${_CMAKE_CUDA_EXTRA_FLAGS} <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> ${CMAKE_CUDA_COMPILE_OPTIONS_PIC} ${_CMAKE_CUDA_EXTRA_DEVICE_LINK_FLAGS} -shared -dlink <OBJECTS> -o <TARGET> <LINK_LIBRARIES>${__IMPLICIT_DLINK_FLAGS}")
endif()

# Used when device linking is handled by CMake.
if(NOT CMAKE_CUDA_DEVICE_LINK_COMPILE)
  set(CMAKE_CUDA_DEVICE_LINK_COMPILE "<CMAKE_CUDA_COMPILER> ${_CMAKE_CUDA_EXTRA_FLAGS} <FLAGS> <LINK_FLAGS> -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__ -D__NV_EXTRA_INITIALIZATION=\"\" -D__NV_EXTRA_FINALIZATION=\"\" -DREGISTERLINKBINARYFILE=\\\"<REGISTER_FILE>\\\" -DFATBINFILE=\\\"<FATBINARY>\\\" ${_CMAKE_COMPILE_AS_CUDA_FLAG} -c \"${CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT}/bin/crt/link.stub\" -o <OBJECT>")
endif()

unset(__IMPLICIT_DLINK_FLAGS)

set(CMAKE_CUDA_USE_LINKER_INFORMATION TRUE)

set(CMAKE_CUDA_INFORMATION_LOADED 1)
