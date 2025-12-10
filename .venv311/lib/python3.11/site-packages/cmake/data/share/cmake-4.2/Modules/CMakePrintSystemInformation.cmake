# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CMakePrintSystemInformation
---------------------------

This module can be used for diagnostics to print system information.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CMakePrintSystemInformation)

Examples
^^^^^^^^

For example, including this module in a project:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  project(Example)

  # ...

  include(CMakePrintSystemInformation)

prints various CMake variables::

  CMAKE_SYSTEM is Linux-6.11.0-17-generic Linux 6.11.0-17-generic x86_64
  CMAKE_SYSTEM file is Platform/Linux
  CMAKE_C_COMPILER is /usr/bin/cc
  CMAKE_CXX_COMPILER is /usr/bin/c++
  CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS is -shared
  CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS is -shared
  ...
#]=======================================================================]

message("CMAKE_SYSTEM is ${CMAKE_SYSTEM} ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION} ${CMAKE_SYSTEM_PROCESSOR}")
message("CMAKE_SYSTEM file is ${CMAKE_SYSTEM_INFO_FILE}")
message("CMAKE_C_COMPILER is ${CMAKE_C_COMPILER}")
message("CMAKE_CXX_COMPILER is ${CMAKE_CXX_COMPILER}")


message("CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS is ${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS}")
message("CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS is ${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS}")
message("CMAKE_DL_LIBS is ${CMAKE_DL_LIBS}")
message("CMAKE_SHARED_LIBRARY_PREFIX is ${CMAKE_SHARED_LIBRARY_PREFIX}")
message("CMAKE_SHARED_LIBRARY_SUFFIX is ${CMAKE_SHARED_LIBRARY_SUFFIX}")
message("CMAKE_C_COMPILER_ID = ${CMAKE_C_COMPILER_ID}")
message("CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")

message("CMAKE_CXX_CREATE_SHARED_LIBRARY is ${CMAKE_CXX_CREATE_SHARED_LIBRARY}")
message("CMAKE_CXX_CREATE_SHARED_MODULE is ${CMAKE_CXX_CREATE_SHARED_MODULE}")
message("CMAKE_CXX_CREATE_STATIC_LIBRARY is ${CMAKE_CXX_CREATE_STATIC_LIBRARY}")
message("CMAKE_CXX_COMPILE_OBJECT is ${CMAKE_CXX_COMPILE_OBJECT}")
message("CMAKE_CXX_LINK_EXECUTABLE ${CMAKE_CXX_LINK_EXECUTABLE}")

message("CMAKE_C_CREATE_SHARED_LIBRARY is ${CMAKE_C_CREATE_SHARED_LIBRARY}")
message("CMAKE_C_CREATE_SHARED_MODULE is ${CMAKE_C_CREATE_SHARED_MODULE}")
message("CMAKE_C_CREATE_STATIC_LIBRARY is ${CMAKE_C_CREATE_STATIC_LIBRARY}")
message("CMAKE_C_COMPILE_OBJECT is ${CMAKE_C_COMPILE_OBJECT}")
message("CMAKE_C_LINK_EXECUTABLE ${CMAKE_C_LINK_EXECUTABLE}")

message("CMAKE_SYSTEM_AND_CXX_COMPILER_INFO_FILE ${CMAKE_SYSTEM_AND_CXX_COMPILER_INFO_FILE}")
message("CMAKE_SYSTEM_AND_C_COMPILER_INFO_FILE ${CMAKE_SYSTEM_AND_C_COMPILER_INFO_FILE}")
