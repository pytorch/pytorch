# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CMakeFindPackageMode
--------------------

This module is executed by ``cmake`` when invoked with the
:ref:`--find-package <Find-Package Tool Mode>` option to locate the requested
package.

.. note::

  This is internal module and is not meant to be included directly in the
  project.  For usage details, refer to the :ref:`--find-package
  <Find-Package Tool Mode>` documentation.
#]=======================================================================]

if(NOT NAME)
  message(FATAL_ERROR "Name of the package to be searched not specified. Set the CMake variable NAME, e.g. -DNAME=JPEG .")
endif()

if(NOT COMPILER_ID)
  message(FATAL_ERROR "COMPILER_ID argument not specified. In doubt, use GNU.")
endif()

if(NOT LANGUAGE)
  message(FATAL_ERROR "LANGUAGE argument not specified. Use C, CXX or Fortran.")
endif()

if(NOT MODE)
  message(FATAL_ERROR "MODE argument not specified. Use either EXIST, COMPILE or LINK.")
endif()

# require the current version. If we don't do this, Platforms/CYGWIN.cmake complains because
# it doesn't know whether it should set WIN32 or not:
cmake_minimum_required(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION} )

macro(ENABLE_LANGUAGE)
  # disable the enable_language() command, otherwise --find-package breaks on Windows.
  # On Windows, enable_language(RC) is called in the platform files unconditionally.
  # But in --find-package mode, we don't want (and can't) enable any language.
endmacro()

set(CMAKE_PLATFORM_INFO_DIR ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY})

include(CMakeDetermineSystem)

# short-cut some tests on Darwin, see Darwin-GNU.cmake:
if("${CMAKE_SYSTEM_NAME}" MATCHES Darwin  AND  "${COMPILER_ID}" MATCHES GNU)
  set(CMAKE_${LANGUAGE}_SYSROOT_FLAG "")
  set(CMAKE_${LANGUAGE}_OSX_DEPLOYMENT_TARGET_FLAG "")
endif()

include(CMakeSystemSpecificInitialize)

# Also load the system specific file, which sets up e.g. the search paths.
# This makes the FIND_XXX() calls work much better
include(CMakeSystemSpecificInformation)

if(UNIX)

  # try to guess whether we have a 64bit system, if it has not been set
  # from the outside
  if(NOT CMAKE_SIZEOF_VOID_P)
    set(CMAKE_SIZEOF_VOID_P 4)
    if(EXISTS ${CMAKE_SYSROOT}/usr/lib64)
      set(CMAKE_SIZEOF_VOID_P 8)
    else()
      # use the file utility to check whether itself is 64 bit:
      find_program(FILE_EXECUTABLE file)
      if(FILE_EXECUTABLE)
        get_filename_component(FILE_ABSPATH "${FILE_EXECUTABLE}" ABSOLUTE)
        execute_process(COMMAND "${FILE_ABSPATH}" "${FILE_ABSPATH}" OUTPUT_VARIABLE fileOutput ERROR_QUIET)
        if("${fileOutput}" MATCHES "64-bit")
          set(CMAKE_SIZEOF_VOID_P 8)
        endif()
      endif()
    endif()
  endif()

  # guess Debian multiarch if it has not been set:
  if(EXISTS /etc/debian_version)
    if(NOT CMAKE_${LANGUAGE}_LIBRARY_ARCHITECTURE )
      file(GLOB filesInLib RELATIVE /lib /lib/*-linux-gnu* )
      foreach(file ${filesInLib})
        if("${file}" MATCHES "${CMAKE_LIBRARY_ARCHITECTURE_REGEX}")
          set(CMAKE_${LANGUAGE}_LIBRARY_ARCHITECTURE ${file})
          break()
        endif()
      endforeach()
    endif()
    if(NOT CMAKE_LIBRARY_ARCHITECTURE)
      set(CMAKE_LIBRARY_ARCHITECTURE ${CMAKE_${LANGUAGE}_LIBRARY_ARCHITECTURE})
    endif()
  endif()

endif()

set(CMAKE_${LANGUAGE}_COMPILER "dummy")
set(CMAKE_${LANGUAGE}_COMPILER_ID "${COMPILER_ID}")
include(CMake${LANGUAGE}Information)


function(set_compile_flags_var _packageName)
  string(TOUPPER "${_packageName}" PACKAGE_NAME)
  # Check the following variables:
  # FOO_INCLUDE_DIRS
  # Foo_INCLUDE_DIRS
  # FOO_INCLUDES
  # Foo_INCLUDES
  # FOO_INCLUDE_DIR
  # Foo_INCLUDE_DIR
  set(includes)
  if(DEFINED ${_packageName}_INCLUDE_DIRS)
    set(includes ${_packageName}_INCLUDE_DIRS)
  elseif(DEFINED ${PACKAGE_NAME}_INCLUDE_DIRS)
    set(includes ${PACKAGE_NAME}_INCLUDE_DIRS)
  elseif(DEFINED ${_packageName}_INCLUDES)
    set(includes ${_packageName}_INCLUDES)
  elseif(DEFINED ${PACKAGE_NAME}_INCLUDES)
    set(includes ${PACKAGE_NAME}_INCLUDES)
  elseif(DEFINED ${_packageName}_INCLUDE_DIR)
    set(includes ${_packageName}_INCLUDE_DIR)
  elseif(DEFINED ${PACKAGE_NAME}_INCLUDE_DIR)
    set(includes ${PACKAGE_NAME}_INCLUDE_DIR)
  endif()

  set(PACKAGE_INCLUDE_DIRS "${${includes}}" PARENT_SCOPE)

  # Check the following variables:
  # FOO_DEFINITIONS
  # Foo_DEFINITIONS
  set(definitions)
  if(DEFINED ${_packageName}_DEFINITIONS)
    set(definitions ${_packageName}_DEFINITIONS)
  elseif(DEFINED ${PACKAGE_NAME}_DEFINITIONS)
    set(definitions ${PACKAGE_NAME}_DEFINITIONS)
  endif()

  set(PACKAGE_DEFINITIONS  "${${definitions}}" )

endfunction()


function(set_link_flags_var _packageName)
  string(TOUPPER "${_packageName}" PACKAGE_NAME)
  # Check the following variables:
  # FOO_LIBRARIES
  # Foo_LIBRARIES
  # FOO_LIBS
  # Foo_LIBS
  set(libs)
  if(DEFINED ${_packageName}_LIBRARIES)
    set(libs ${_packageName}_LIBRARIES)
  elseif(DEFINED ${PACKAGE_NAME}_LIBRARIES)
    set(libs ${PACKAGE_NAME}_LIBRARIES)
  elseif(DEFINED ${_packageName}_LIBS)
    set(libs ${_packageName}_LIBS)
  elseif(DEFINED ${PACKAGE_NAME}_LIBS)
    set(libs ${PACKAGE_NAME}_LIBS)
  endif()

  set(PACKAGE_LIBRARIES "${${libs}}" PARENT_SCOPE )

endfunction()


find_package("${NAME}" QUIET)

set(PACKAGE_FOUND FALSE)

string(TOUPPER "${NAME}" UPPERCASE_NAME)

if(${NAME}_FOUND  OR  ${UPPERCASE_NAME}_FOUND)
  set(PACKAGE_FOUND TRUE)

  if("${MODE}" STREQUAL "EXIST")
    # do nothing
  elseif("${MODE}" STREQUAL "COMPILE")
    set_compile_flags_var(${NAME})
  elseif("${MODE}" STREQUAL "LINK")
    set_link_flags_var(${NAME})
  else()
    message(FATAL_ERROR "Invalid mode argument ${MODE} given.")
  endif()

endif()

set(PACKAGE_QUIET ${SILENT} )
