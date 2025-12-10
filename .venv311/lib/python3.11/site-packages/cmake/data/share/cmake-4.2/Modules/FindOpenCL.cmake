# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindOpenCL
----------

.. versionadded:: 3.1

Finds Open Computing Language (OpenCL):

.. code-block:: cmake

  find_package(OpenCL [<version>] [...])

OpenCL is a framework for writing programs that execute across heterogeneous
platforms, such as CPUs, GPUs, and other accelerators.

.. versionadded:: 3.24
  Detection of OpenCL 3.0.

.. versionadded:: 3.10
  Detection of OpenCL 2.1 and 2.2.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``OpenCL::OpenCL``
  .. versionadded:: 3.7

  Target encapsulating the OpenCL usage requirements, available if OpenCL
  has been found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``OpenCL_FOUND``
  Boolean indicating whether (the requested version of) OpenCL was found.

``OpenCL_VERSION``
  .. versionadded:: 4.2

  Highest supported OpenCL version found in form of ``<major>.<minor>``
  (e.g., ``1.2``).

``OpenCL_VERSION_MAJOR``
  The major version of the OpenCL implementation.

``OpenCL_VERSION_MINOR``
  The minor version of the OpenCL implementation.

``OpenCL_INCLUDE_DIRS``
  Include directories needed to use OpenCL.

``OpenCL_LIBRARIES``
  Libraries needed to link to OpenCL.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OpenCL_INCLUDE_DIR``
  The OpenCL include directory.

``OpenCL_LIBRARY``
  The path to the OpenCL library.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OpenCL_VERSION_STRING``
  .. deprecated:: 4.2
    Use ``OpenCL_VERSION``, which has the same value.

  Highest supported OpenCL version found in form of ``<major>.<minor>``.

Examples
^^^^^^^^

Finding OpenCL and linking it to a project target:

.. code-block:: cmake

  find_package(OpenCL)
  target_link_libraries(project_target PRIVATE OpenCL::OpenCL)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0140 NEW)

set(_OPENCL_x86 "(x86)")

function(_FIND_OPENCL_VERSION)
  include(CheckIncludeFiles)
  include(CheckSymbolExists)
  include(CMakePushCheckState)

  cmake_push_check_state()

  set(CMAKE_REQUIRED_QUIET ${OpenCL_FIND_QUIETLY})
  set(CMAKE_REQUIRED_INCLUDES "${OpenCL_INCLUDE_DIR}")

  check_include_files(OpenCL/cl.h OpenCL_HAVE_OPENCL_CL_H)

  foreach(VERSION "3_0" "2_2" "2_1" "2_0" "1_2" "1_1" "1_0")
    if(OpenCL_HAVE_OPENCL_CL_H)
      check_symbol_exists(
        CL_VERSION_${VERSION}
        "OpenCL/cl.h"
        OPENCL_VERSION_${VERSION})
    else()
      check_symbol_exists(
        CL_VERSION_${VERSION}
        "CL/cl.h"
        OPENCL_VERSION_${VERSION})
    endif()

    if(OPENCL_VERSION_${VERSION})
      string(REPLACE "_" "." VERSION "${VERSION}")
      set(OpenCL_VERSION ${VERSION})
      set(OpenCL_VERSION_STRING "${OpenCL_VERSION}")
      string(REGEX MATCHALL "[0-9]+" version_components "${VERSION}")
      list(GET version_components 0 major_version)
      list(GET version_components 1 minor_version)
      set(OpenCL_VERSION_MAJOR ${major_version})
      set(OpenCL_VERSION_MINOR ${minor_version})
      break()
    endif()
  endforeach()
  cmake_pop_check_state()

  return(
    PROPAGATE
      OpenCL_VERSION
      OpenCL_VERSION_MAJOR
      OpenCL_VERSION_MINOR
      OpenCL_VERSION_STRING
  )
endfunction()

find_path(OpenCL_INCLUDE_DIR
  NAMES
    CL/cl.h OpenCL/cl.h
  PATHS
    ENV "PROGRAMFILES(X86)"
    ENV "PROGRAMFILES"
    $ENV{PROGRAMFILES${_OPENCL_x86}}/OpenCLHeaders
    $ENV{PROGRAMFILES}/OpenCLHeaders
    ENV AMDAPPSDKROOT
    ENV INTELOCLSDKROOT
    ENV NVSDKCOMPUTE_ROOT
    ENV CUDA_PATH
    ENV ATISTREAMSDKROOT
    ENV OCL_ROOT
    /usr/local/cuda
    /opt/cuda
  PATH_SUFFIXES
    include
    OpenCL/common/inc
    "AMD APP/include")

_FIND_OPENCL_VERSION()

if(WIN32)
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    find_library(OpenCL_LIBRARY
      NAMES OpenCL
      PATHS
        ENV "PROGRAMFILES(X86)"
        ENV "PROGRAMFILES"
        $ENV{PROGRAMFILES${_OPENCL_x86}}/OpenCL-ICD-Loader
        $ENV{PROGRAMFILES}/OpenCL-ICD-Loader
        ENV AMDAPPSDKROOT
        ENV INTELOCLSDKROOT
        ENV CUDA_PATH
        ENV NVSDKCOMPUTE_ROOT
        ENV ATISTREAMSDKROOT
        ENV OCL_ROOT
      PATH_SUFFIXES
        "AMD APP/lib/x86"
        lib/x86
        lib/Win32
        OpenCL/common/lib/Win32)
  elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    find_library(OpenCL_LIBRARY
      NAMES OpenCL
      PATHS
        ENV "PROGRAMFILES(X86)"
        ENV "PROGRAMFILES"
        $ENV{PROGRAMFILES${_OPENCL_x86}}/OpenCL-ICD-Loader
        $ENV{PROGRAMFILES}/OpenCL-ICD-Loader
        ENV AMDAPPSDKROOT
        ENV INTELOCLSDKROOT
        ENV CUDA_PATH
        ENV NVSDKCOMPUTE_ROOT
        ENV ATISTREAMSDKROOT
        ENV OCL_ROOT
      PATH_SUFFIXES
        "AMD APP/lib/x86_64"
        lib/x86_64
        lib/x64
        lib
        OpenCL/common/lib/x64)
  endif()
else()
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    find_library(OpenCL_LIBRARY
      NAMES OpenCL
      PATHS
        ENV AMDAPPSDKROOT
        ENV CUDA_PATH
        /usr/local/cuda
        /opt/cuda
      PATH_SUFFIXES
        lib/x86
        lib)
  elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    find_library(OpenCL_LIBRARY
      NAMES OpenCL
      PATHS
        ENV AMDAPPSDKROOT
        ENV CUDA_PATH
        /usr/local/cuda
        /opt/cuda
      PATH_SUFFIXES
        lib/x86_64
        lib/x64
        lib
        lib64)
  endif()
endif()

unset(_OPENCL_x86)

set(OpenCL_LIBRARIES ${OpenCL_LIBRARY})
set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  OpenCL
  REQUIRED_VARS OpenCL_LIBRARY OpenCL_INCLUDE_DIR
  VERSION_VAR OpenCL_VERSION
)

mark_as_advanced(
  OpenCL_INCLUDE_DIR
  OpenCL_LIBRARY)

if(OpenCL_FOUND AND NOT TARGET OpenCL::OpenCL)
  if(OpenCL_LIBRARY MATCHES "/([^/]+)\\.framework$")
    add_library(OpenCL::OpenCL INTERFACE IMPORTED)
    set_target_properties(OpenCL::OpenCL PROPERTIES
      INTERFACE_LINK_LIBRARIES "${OpenCL_LIBRARY}")
  else()
    add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
    set_target_properties(OpenCL::OpenCL PROPERTIES
      IMPORTED_LOCATION "${OpenCL_LIBRARY}")
  endif()
  set_target_properties(OpenCL::OpenCL PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${OpenCL_INCLUDE_DIRS}")
endif()

cmake_policy(POP)
