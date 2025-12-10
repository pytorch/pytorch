# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindSWIG
--------

Finds the installed Simplified Wrapper and Interface Generator (SWIG_)
executable and determines its version:

.. code-block:: cmake

  find_package(SWIG [<version>] [COMPONENTS <langs>...] [...])

.. versionadded:: 3.19
  Support for specifying version range when calling the :command:`find_package`
  command.  When a version is requested, it can be specified as a single
  value as before, and now also a version range can be used.  For a detailed
  description of version range usage and capabilities, refer to the
  :command:`find_package` command.

Components
^^^^^^^^^^

.. versionadded:: 3.18

This module supports optional components to specify target languages.

If a ``COMPONENTS`` or ``OPTIONAL_COMPONENTS`` argument is given to the
:command:`find_package` command, it will also determine supported target
languages.

.. code-block:: cmake

  find_package(SWIG [COMPONENTS <langs>...] [OPTIONAL_COMPONENTS <langs>...])

Any ``COMPONENTS`` given to ``find_package()`` should be the names of
supported target languages as provided to the ``LANGUAGE`` argument of
:command:`swig_add_library`, such as ``python`` or ``perl5``.  Language
names *must* be lowercase.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``SWIG_FOUND``
  Boolean indicating whether (the requested version of) SWIG and any required
  components were found on the system.
``SWIG_VERSION``
  SWIG executable version (result of ``swig -version``).
``SWIG_<lang>_FOUND``
  If ``COMPONENTS`` or ``OPTIONAL_COMPONENTS`` are requested, each available
  target language ``<lang>`` (lowercase) will be set to TRUE.
``SWIG_DIR``
  Path to the installed SWIG ``Lib`` directory (result of ``swig -swiglib``).

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``SWIG_EXECUTABLE``
  The path to the SWIG executable.

  This executable is used to retrieve all information for this module. It can
  be also manually set to change the version to be found from the command line.

Examples
^^^^^^^^

Example usage requiring SWIG 4.0 or higher and Python language support, with
optional Fortran support:

.. code-block:: cmake

   find_package(SWIG 4.0 COMPONENTS python OPTIONAL_COMPONENTS fortran)
   if(SWIG_FOUND)
     message("SWIG found: ${SWIG_EXECUTABLE}")
     if(NOT SWIG_fortran_FOUND)
       message(WARNING "SWIG Fortran bindings cannot be generated")
     endif()
   endif()

This module is commonly used in conjunction with the :module:`UseSWIG` module:

.. code-block:: cmake

  find_package(SWIG COMPONENTS python)
  if(SWIG_FOUND)
    include(UseSWIG)

    swig_add_library(mymod LANGUAGE python SOURCES mymod.i)
  endif()

See Also
^^^^^^^^

* The :module:`UseSWIG` module to use SWIG in CMake.

.. _SWIG: https://swig.org
#]=======================================================================]

include(FindPackageHandleStandardArgs)

function(_swig_get_version _swig_executable _swig_version)
  unset(${_swig_version} PARENT_SCOPE)
  # Determine SWIG version
  execute_process(COMMAND "${_swig_executable}" -version
    OUTPUT_VARIABLE _swig_output
    ERROR_VARIABLE _swig_output
    RESULT_VARIABLE _swig_result)
  if(_swig_result)
    set_property (CACHE _SWIG_REASON_FAILURE PROPERTY VALUE "Cannot use the executable \"${_swig_executable}\"")
    if (_swig_output)
      set_property (CACHE _SWIG_REASON_FAILURE APPEND_STRING PROPERTY VALUE ": ${_swig_output}")
    endif()
  else()
    string(REGEX REPLACE ".*SWIG Version[^0-9.]*\([0-9.]+\).*" "\\1"
                         _swig_output "${_swig_output}")
    set(${_swig_version} ${_swig_output} PARENT_SCOPE)
  endif()
endfunction()

function(_swig_validate_find_executable status executable)
  _swig_get_version("${executable}" _swig_find_version)
  if(NOT _swig_find_version)
    # executable is unusable
    set (${status} FALSE PARENT_SCOPE)
    return()
  endif()
  if(NOT SWIG_FIND_VERSION)
    return()
  endif()

  find_package_check_version(${_swig_find_version}  _swig_version_is_valid HANDLE_VERSION_RANGE)
  if(_swig_version_is_valid)
    unset(_SWIG_REASON_FAILURE CACHE)
  else()
    set (${status} FALSE PARENT_SCOPE)
    set_property (CACHE _SWIG_REASON_FAILURE PROPERTY VALUE "Could NOT find SWIG: Found unsuitable version \"${_swig_find_version}\" for the executable \"${executable}\"")
  endif()
endfunction()

unset (_SWIG_REASON_FAILURE)
set (_SWIG_REASON_FAILURE CACHE INTERNAL "SWIG reason failure")

# compute list of possible names
unset (_SWIG_NAMES)
if (SWIG_FIND_VERSION_RANGE)
  foreach (_SWIG_MAJOR IN ITEMS 4 3 2)
    if (_SWIG_MAJOR VERSION_GREATER_EQUAL SWIG_FIND_VERSION_MIN_MAJOR
        AND ((SWIG_FIND_VERSION_RANGE_MAX STREQUAL "INCLUDE" AND _SWIG_MAJOR VERSION_LESS_EQUAL SWIG_FIND_VERSION_MAX)
        OR (SWIG_FIND_VERSION_RANGE_MAX STREQUAL "EXCLUDE" AND _SWIG_MAJOR VERSION_LESS SWIG_FIND_VERSION_MAX)))
      list (APPEND _SWIG_NAMES swig${_SWIG_MAJOR}.0)
    endif()
  endforeach()
elseif(SWIG_FIND_VERSION)
  if (SWIG_FIND_VERSION_EXACT)
    set(_SWIG_NAMES swig${SWIG_FIND_VERSION_MAJOR}.0)
  else()
    foreach (_SWIG_MAJOR IN ITEMS 4 3 2)
      if (_SWIG_MAJOR VERSION_GREATER_EQUAL SWIG_FIND_VERSION_MAJOR)
        list (APPEND _SWIG_NAMES swig${_SWIG_MAJOR}.0)
      endif()
    endforeach()
  endif()
else()
  set (_SWIG_NAMES swig4.0 swig3.0 swig2.0)
endif()
if (NOT _SWIG_NAMES)
  # try to find any version
  set (_SWIG_NAMES swig4.0 swig3.0 swig2.0)
endif()

find_program(SWIG_EXECUTABLE NAMES ${_SWIG_NAMES} swig NAMES_PER_DIR
                             VALIDATOR _swig_validate_find_executable)
unset(_SWIG_NAMES)

if(SWIG_EXECUTABLE AND NOT SWIG_DIR)
  # Find default value for SWIG library directory
  execute_process(COMMAND "${SWIG_EXECUTABLE}" -swiglib
    OUTPUT_VARIABLE _swig_output
    ERROR_VARIABLE _swig_error
    RESULT_VARIABLE _swig_result)

  if(_swig_result)
    set(_msg "Command \"${SWIG_EXECUTABLE} -swiglib\" failed with output:\n${_swig_error}")
    if(SWIG_FIND_REQUIRED)
      message(SEND_ERROR "${_msg}")
    else()
      message(STATUS "${_msg}")
    endif()
    unset(_msg)
  else()
    string(REGEX REPLACE "[\n\r]+" ";" _SWIG_LIB ${_swig_output})
  endif()

  # Find SWIG library directory
  find_path(SWIG_DIR swig.swg PATHS ${_SWIG_LIB} NO_CMAKE_FIND_ROOT_PATH)
  unset(_SWIG_LIB)
endif()

if(SWIG_EXECUTABLE AND SWIG_DIR AND NOT SWIG_VERSION)
  # Determine SWIG version
  _swig_get_version("${SWIG_EXECUTABLE}" _swig_output)
  set(SWIG_VERSION ${_swig_output} CACHE STRING "Swig version" FORCE)
endif()

if(SWIG_EXECUTABLE AND SWIG_FIND_COMPONENTS)
  execute_process(COMMAND "${SWIG_EXECUTABLE}" -help
    OUTPUT_VARIABLE _swig_output
    ERROR_VARIABLE _swig_error
    RESULT_VARIABLE _swig_result)
  if(_swig_result)
    message(SEND_ERROR "Command \"${SWIG_EXECUTABLE} -help\" failed with output:\n${_swig_error}")
  else()
    string(REPLACE "\n" ";" _swig_output "${_swig_output}")
    foreach(SWIG_line IN LISTS _swig_output)
      if(SWIG_line MATCHES "-([A-Za-z0-9_]+) +- *Generate.*wrappers")
        set(SWIG_${CMAKE_MATCH_1}_FOUND TRUE)
      endif()
    endforeach()
  endif()
endif()

find_package_handle_standard_args(
  SWIG HANDLE_COMPONENTS
  REQUIRED_VARS SWIG_EXECUTABLE SWIG_DIR
  VERSION_VAR SWIG_VERSION
  HANDLE_VERSION_RANGE
  FAIL_MESSAGE "${_SWIG_REASON_FAILURE}")

unset(_swig_output)
unset(_swig_error)
unset(_swig_result)

unset(_SWIG_REASON_FAILURE CACHE)

if(SWIG_FOUND)
  set(SWIG_USE_FILE "${CMAKE_CURRENT_LIST_DIR}/UseSWIG.cmake")
endif()

mark_as_advanced(SWIG_DIR SWIG_VERSION SWIG_EXECUTABLE)
