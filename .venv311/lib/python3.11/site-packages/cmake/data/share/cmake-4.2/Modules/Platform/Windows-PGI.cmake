# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__WINDOWS_COMPILER_PGI)
  return()
endif()
set(__WINDOWS_COMPILER_PGI 1)

# PGI on Windows doesn't support parallel compile processes
if(NOT DEFINED CMAKE_JOB_POOL_LINK OR NOT DEFINED CMAKE_JOB_POOL_COMPILE OR NOT DEFINED CMAKE_JOB_POOL_PRECOMPILE_HEADER)
  set(CMAKE_JOB_POOL_LINK PGITaskPool)
  set(CMAKE_JOB_POOL_COMPILE PGITaskPool)
  set(CMAKE_JOB_POOL_PRECOMPILE_HEADER PGITaskPool)
  get_property(_pgijp GLOBAL PROPERTY JOB_POOLS)
  if(NOT _pgijp MATCHES "PGITaskPool=")
      set_property(GLOBAL APPEND PROPERTY JOB_POOLS PGITaskPool=1)
  endif()
  unset(_pgijp)
endif()

set(CMAKE_SUPPORT_WINDOWS_EXPORT_ALL_SYMBOLS 1)
set(CMAKE_LINK_DEF_FILE_FLAG "-def:")
# The link flags for PGI are the raw filename to add a file
# and the UNIX -L syntax to link directories.
set(CMAKE_LINK_LIBRARY_FLAG "")
set(CMAKE_LINK_STARTFILE "pgimain[mx][xpt]+[.]obj")

# Default to Debug builds, mirroring Windows-MSVC behavior
set(CMAKE_BUILD_TYPE_INIT Debug)

if(CMAKE_VERBOSE_MAKEFILE)
  set(CMAKE_CL_NOLOGO)
else()
  set(CMAKE_CL_NOLOGO "/nologo")
endif()

macro(__windows_compiler_pgi lang)
  set(CMAKE_${lang}_LINK_DEF_FILE_FLAG "-def:")

  # Shared library compile and link rules.
  set(CMAKE_${lang}_CREATE_STATIC_LIBRARY "lib ${CMAKE_CL_NOLOGO} <LINK_FLAGS> /out:<TARGET> <OBJECTS> ")
  set(CMAKE_${lang}_CREATE_SHARED_LIBRARY "<CMAKE_${lang}_COMPILER> ${CMAKE_START_TEMP_FILE} -Mmakedll -implib:<TARGET_IMPLIB> -Xlinker -pdb:<TARGET_PDB> -Xlinker -version:<TARGET_VERSION_MAJOR>.<TARGET_VERSION_MINOR> <LINK_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES> ${CMAKE_END_TEMP_FILE}")
  set(CMAKE_${lang}_CREATE_SHARED_MODULE "${CMAKE_${lang}_CREATE_SHARED_LIBRARY}")
  set(CMAKE_${lang}_LINK_EXECUTABLE "<CMAKE_${lang}_COMPILER> ${CMAKE_START_TEMP_FILE} -implib:<TARGET_IMPLIB> -Xlinker -pdb:<TARGET_PDB> -Xlinker -version:<TARGET_VERSION_MAJOR>.<TARGET_VERSION_MINOR> <FLAGS> <LINK_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES> ${CMAKE_END_TEMP_FILE}")

  if("${lang}" MATCHES "C|CXX")
    set(CMAKE_${lang}_STANDARD_LIBRARIES_INIT "kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib")
  endif()
endmacro()
