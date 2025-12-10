# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindHg
------

Finds the Mercurial command-line client executable (``hg``) and provides a
command for extracting information from a Mercurial working copy:

.. code-block:: cmake

  find_package(Hg [<version>] [...])

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Hg_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the (requested version of) Mercurial client was
  found.

``Hg_VERSION``
  .. versionadded:: 4.2

  The version of Mercurial found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``HG_EXECUTABLE``
  Absolute path to the Mercurial command-line client (``hg``).

Commands
^^^^^^^^

This module provides the following command when Mercurial client (``hg``) is
found:

.. command:: Hg_WC_INFO

  .. versionadded:: 3.1

  Extracts information of a Mercurial working copy:

  .. code-block:: cmake

    Hg_WC_INFO(<dir> <var-prefix>)

  This command defines the following variables if running Mercurial client on
  working copy located at a given location ``<dir>`` succeeds; otherwise a
  ``SEND_ERROR`` message is generated:

  ``<var-prefix>_WC_CHANGESET``
    Current changeset.
  ``<var-prefix>_WC_REVISION``
    Current revision.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``HG_FOUND``
  .. deprecated:: 4.2
    Use ``Hg_FOUND``, which has the same value.

  Boolean indicating whether the (requested version of) Mercurial client was
  found.

``HG_VERSION_STRING``
  .. deprecated:: 4.2
    Use ``Hg_VERSION``, which has the same value.

  The version of Mercurial found.

Examples
^^^^^^^^

Finding the Mercurial client and retrieving information about the current
project's working copy:

.. code-block:: cmake

  find_package(Hg)
  if(Hg_FOUND)
    Hg_WC_INFO(${PROJECT_SOURCE_DIR} Project)
    message("Current revision is ${Project_WC_REVISION}")
    message("Current changeset is ${Project_WC_CHANGESET}")
  endif()
#]=======================================================================]

find_program(HG_EXECUTABLE
  NAMES hg
  PATHS
    [HKEY_LOCAL_MACHINE\\Software\\TortoiseHG]
  PATH_SUFFIXES Mercurial
  DOC "hg command line client"
  )
mark_as_advanced(HG_EXECUTABLE)

if(HG_EXECUTABLE)
  set(_saved_lc_all "$ENV{LC_ALL}")
  set(ENV{LC_ALL} "C")

  set(_saved_language "$ENV{LANGUAGE}")
  set(ENV{LANGUAGE})

  execute_process(COMMAND ${HG_EXECUTABLE} --version
                  OUTPUT_VARIABLE hg_version
                  ERROR_QUIET
                  RESULT_VARIABLE hg_result
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(ENV{LC_ALL} ${_saved_lc_all})
  set(ENV{LANGUAGE} ${_saved_language})

  if(hg_result MATCHES "is not a valid Win32 application")
    set_property(CACHE HG_EXECUTABLE PROPERTY VALUE "HG_EXECUTABLE-NOTFOUND")
  endif()
  if(hg_version MATCHES "^Mercurial Distributed SCM \\(version ([0-9][^)]*)\\)")
    set(Hg_VERSION "${CMAKE_MATCH_1}")
    set(HG_VERSION_STRING "${Hg_VERSION}")
  endif()
  unset(hg_version)

  macro(HG_WC_INFO dir prefix)
    execute_process(COMMAND ${HG_EXECUTABLE} id -i -n
      WORKING_DIRECTORY ${dir}
      RESULT_VARIABLE hg_id_result
      ERROR_VARIABLE hg_id_error
      OUTPUT_VARIABLE ${prefix}_WC_DATA
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT ${hg_id_result} EQUAL 0)
      message(SEND_ERROR "Command \"${HG_EXECUTABLE} id -n\" in directory ${dir} failed with output:\n${hg_id_error}")
    endif()

    string(REGEX REPLACE "([0-9a-f]+)\\+? [0-9]+\\+?" "\\1" ${prefix}_WC_CHANGESET ${${prefix}_WC_DATA})
    string(REGEX REPLACE "[0-9a-f]+\\+? ([0-9]+)\\+?" "\\1" ${prefix}_WC_REVISION ${${prefix}_WC_DATA})
  endmacro()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Hg
                                  REQUIRED_VARS HG_EXECUTABLE
                                  VERSION_VAR Hg_VERSION)
