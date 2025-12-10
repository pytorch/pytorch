# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindSubversion
--------------

Finds a Subversion command-line client executable (``svn``) and provides
commands for extracting information from a Subversion working copy:

.. code-block:: cmake

  find_package(Subversion [<version>] [...])

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Subversion_FOUND``
  Boolean indicating whether the (requested version of) Subversion
  command-line client was found.

``Subversion_VERSION``
  .. versionadded:: 4.2

  Version of the ``svn`` command-line client found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Subversion_SVN_EXECUTABLE``
  Path to the ``svn`` command-line client.

Commands
^^^^^^^^

This module provides the following commands if the Subversion command-line
client is found:

.. command:: Subversion_WC_INFO

  Extracts information from a Subversion working copy located at a specified
  directory:

  .. code-block:: cmake

    Subversion_WC_INFO(<dir> <var-prefix> [IGNORE_SVN_FAILURE])

  This command defines the following variables if running Subversion's ``info``
  subcommand on ``<dir>`` succeeds; otherwise a ``SEND_ERROR`` message is
  generated:

  ``<var-prefix>_WC_URL``
    URL of the repository (at ``<dir>``).
  ``<var-prefix>_WC_ROOT``
    Root URL of the repository.
  ``<var-prefix>_WC_REVISION``
    Current revision.
  ``<var-prefix>_WC_LAST_CHANGED_AUTHOR``
    Author of last commit.
  ``<var-prefix>_WC_LAST_CHANGED_DATE``
    Date of last commit.
  ``<var-prefix>_WC_LAST_CHANGED_REV``
    Revision of last commit.
  ``<var-prefix>_WC_INFO``
    Output of the command ``svn info <dir>``

  The options are:

  ``IGNORE_SVN_FAILURE``
    .. versionadded:: 3.13

    When specified, errors from Subversion operation will not trigger a
    ``SEND_ERROR`` message.  In case of an error, the ``<var-prefix>_*``
    variables remain undefined.

.. command:: Subversion_WC_LOG

  Retrieves the log message of the base revision of a Subversion working copy at
  a given location:

  .. code-block:: cmake

    Subversion_WC_LOG(<dir> <var-prefix>)

  This command defines the following variable if running Subversion's ``log``
  subcommand on ``<dir>`` succeeds; otherwise a ``SEND_ERROR`` message is
  generated:

  ``<var-prefix>_LAST_CHANGED_LOG``
    Last log of the base revision of a Subversion working copy located at
    ``<dir>``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``SUBVERSION_FOUND``
  .. deprecated:: 4.2
    Use ``Subversion_FOUND``, which has the same value.

  Boolean indicating whether the (requested version of) Subversion
  command-line client was found.

``Subversion_VERSION_SVN``
  .. deprecated:: 4.2
    Use the ``Subversion_VERSION``.

  Version of the ``svn`` command-line client found.

Examples
^^^^^^^^

Examples: Finding Subversion
""""""""""""""""""""""""""""

Finding Subversion:

.. code-block:: cmake

  find_package(Subversion)

Or, finding Subversion and specifying a minimum required version:

.. code-block:: cmake

  find_package(Subversion 1.4)

Or, finding Subversion and making it required (if not found, processing stops
with an error message):

.. code-block:: cmake

  find_package(Subversion REQUIRED)

Example: Using Subversion
"""""""""""""""""""""""""

Finding Subversion and retrieving information about the current project's
working copy:

.. code-block:: cmake

  find_package(Subversion)
  if(Subversion_FOUND)
    Subversion_WC_INFO(${PROJECT_SOURCE_DIR} Project)
    message("Current revision is ${Project_WC_REVISION}")
    Subversion_WC_LOG(${PROJECT_SOURCE_DIR} Project)
    message("Last changed log is ${Project_LAST_CHANGED_LOG}")
  endif()
#]=======================================================================]

find_program(Subversion_SVN_EXECUTABLE svn
  PATHS
    [HKEY_LOCAL_MACHINE\\Software\\TortoiseSVN;Directory]/bin
  DOC "subversion command line client")
mark_as_advanced(Subversion_SVN_EXECUTABLE)

if(Subversion_SVN_EXECUTABLE)
  # the subversion commands should be executed with the C locale, otherwise
  # the message (which are parsed) may be translated, Alex
  set(_Subversion_SAVED_LC_ALL "$ENV{LC_ALL}")
  set(ENV{LC_ALL} C)

  execute_process(COMMAND ${Subversion_SVN_EXECUTABLE} --version
    OUTPUT_VARIABLE Subversion_VERSION_SVN
    ERROR_VARIABLE _Subversion_VERSION_STDERR
    RESULT_VARIABLE _Subversion_VERSION_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # restore the previous LC_ALL
  set(ENV{LC_ALL} ${_Subversion_SAVED_LC_ALL})

  if(_Subversion_VERSION_RESULT EQUAL 0)
    string(REGEX REPLACE "^(.*\n)?svn, version ([.0-9]+).*"
      "\\2" Subversion_VERSION_SVN "${Subversion_VERSION_SVN}")
  else()
    unset(Subversion_VERSION_SVN)
    if(_Subversion_VERSION_STDERR MATCHES "svn: error: The subversion command line tools are no longer provided by Xcode")
      set(Subversion_SVN_EXECUTABLE Subversion_SVN_EXECUTABLE-NOTFOUND)
    endif()
  endif()

  if(DEFINED Subversion_VERSION_SVN)
    set(Subversion_VERSION "${Subversion_VERSION_SVN}")
  else()
    unset(Subversion_VERSION)
  endif()

  macro(Subversion_WC_INFO dir prefix)

    cmake_parse_arguments(
      "Subversion_WC_INFO"
      "IGNORE_SVN_FAILURE"
      "" ""
      ${ARGN}
    )

    # the subversion commands should be executed with the C locale, otherwise
    # the message (which are parsed) may be translated, Alex
    set(_Subversion_SAVED_LC_ALL "$ENV{LC_ALL}")
    set(ENV{LC_ALL} C)

    execute_process(COMMAND ${Subversion_SVN_EXECUTABLE} info ${dir}
      OUTPUT_VARIABLE ${prefix}_WC_INFO
      ERROR_VARIABLE Subversion_svn_info_error
      RESULT_VARIABLE Subversion_svn_info_result
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(${Subversion_svn_info_result} EQUAL 0)
      string(REGEX REPLACE "^(.*\n)?URL: ([^\n]+).*"
        "\\2" ${prefix}_WC_URL "${${prefix}_WC_INFO}")
      string(REGEX REPLACE "^(.*\n)?Repository Root: ([^\n]+).*"
        "\\2" ${prefix}_WC_ROOT "${${prefix}_WC_INFO}")
      string(REGEX REPLACE "^(.*\n)?Revision: ([^\n]+).*"
        "\\2" ${prefix}_WC_REVISION "${${prefix}_WC_INFO}")
      string(REGEX REPLACE "^(.*\n)?Last Changed Author: ([^\n]+).*"
        "\\2" ${prefix}_WC_LAST_CHANGED_AUTHOR "${${prefix}_WC_INFO}")
      string(REGEX REPLACE "^(.*\n)?Last Changed Rev: ([^\n]+).*"
        "\\2" ${prefix}_WC_LAST_CHANGED_REV "${${prefix}_WC_INFO}")
      string(REGEX REPLACE "^(.*\n)?Last Changed Date: ([^\n]+).*"
        "\\2" ${prefix}_WC_LAST_CHANGED_DATE "${${prefix}_WC_INFO}")
    elseif(NOT Subversion_WC_INFO_IGNORE_SVN_FAILURE)
      message(SEND_ERROR "Command \"${Subversion_SVN_EXECUTABLE} info ${dir}\" failed with output:\n${Subversion_svn_info_error}")
    endif()

    # restore the previous LC_ALL
    set(ENV{LC_ALL} ${_Subversion_SAVED_LC_ALL})

  endmacro()

  macro(Subversion_WC_LOG dir prefix)
    # This macro can block if the certificate is not signed:
    # svn ask you to accept the certificate and wait for your answer
    # This macro requires a svn server network access (Internet most of the time)
    # and can also be slow since it access the svn server
    execute_process(COMMAND
      ${Subversion_SVN_EXECUTABLE} --non-interactive log -r BASE ${dir}
      OUTPUT_VARIABLE ${prefix}_LAST_CHANGED_LOG
      ERROR_VARIABLE Subversion_svn_log_error
      RESULT_VARIABLE Subversion_svn_log_result
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(NOT ${Subversion_svn_log_result} EQUAL 0)
      message(SEND_ERROR "Command \"${Subversion_SVN_EXECUTABLE} log -r BASE ${dir}\" failed with output:\n${Subversion_svn_log_error}")
    endif()
  endmacro()

endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Subversion REQUIRED_VARS Subversion_SVN_EXECUTABLE
                                             VERSION_VAR Subversion_VERSION)

# for compatibility
set(Subversion_SVN_FOUND ${Subversion_FOUND})
