# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindPerl
--------

Finds a Perl interpreter:

.. code-block:: cmake

  find_package(Perl [<version>] [...])

Perl is a general-purpose, interpreted, dynamic programming language.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Perl_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the (requested version of) Perl executable was
  found.

``Perl_VERSION``
  .. versionadded:: 4.2

  The version of Perl found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``PERL_EXECUTABLE``
  Full path to the ``perl`` executable.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``PERL_FOUND``
  .. deprecated:: 4.2
    Use ``Perl_FOUND``, which has the same value.

  Boolean indicating whether the (requested version of) Perl executable was
  found.

``PERL_VERSION_STRING``
  .. deprecated:: 4.2
    Superseded by the ``Perl_VERSION``.

  The version of Perl found.

Examples
^^^^^^^^

Finding the Perl interpreter and executing it in a process:

.. code-block:: cmake

  find_package(Perl)

  if(Perl_FOUND)
    execute_process(COMMAND ${PERL_EXECUTABLE} --help)
  endif()

See Also
^^^^^^^^

* The :module:`FindPerlLibs` to find Perl libraries.
#]=======================================================================]

include(${CMAKE_CURRENT_LIST_DIR}/FindCygwin.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/FindMsys.cmake)

set(PERL_POSSIBLE_BIN_PATHS
  ${CYGWIN_INSTALL_PATH}/bin
  ${MSYS_INSTALL_PATH}/usr/bin
  )

if(WIN32)
  get_filename_component(
    ActivePerl_CurrentVersion
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\ActiveState\\ActivePerl;CurrentVersion]"
    NAME)
  set(PERL_POSSIBLE_BIN_PATHS ${PERL_POSSIBLE_BIN_PATHS}
    "C:/Perl/bin"
    "C:/Strawberry/perl/bin"
    [HKEY_LOCAL_MACHINE\\SOFTWARE\\ActiveState\\ActivePerl\\${ActivePerl_CurrentVersion}]/bin
    )
endif()

find_program(PERL_EXECUTABLE
  NAMES perl
  PATHS ${PERL_POSSIBLE_BIN_PATHS}
  )

if(PERL_EXECUTABLE)
  execute_process(
    COMMAND
      ${PERL_EXECUTABLE} -V:version
      OUTPUT_VARIABLE
        PERL_VERSION_OUTPUT_VARIABLE
      RESULT_VARIABLE
        PERL_VERSION_RESULT_VARIABLE
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT PERL_VERSION_RESULT_VARIABLE AND NOT PERL_VERSION_OUTPUT_VARIABLE MATCHES "^version='UNKNOWN'")
    string(REGEX REPLACE "version='([^']+)'.*" "\\1" Perl_VERSION ${PERL_VERSION_OUTPUT_VARIABLE})
    set(PERL_VERSION_STRING "${Perl_VERSION}")
  else()
    execute_process(
      COMMAND ${PERL_EXECUTABLE} -v
      OUTPUT_VARIABLE PERL_VERSION_OUTPUT_VARIABLE
      RESULT_VARIABLE PERL_VERSION_RESULT_VARIABLE
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT PERL_VERSION_RESULT_VARIABLE AND PERL_VERSION_OUTPUT_VARIABLE MATCHES "This is perl.*[ \\(]v([0-9\\._]+)[ \\)]")
      set(Perl_VERSION "${CMAKE_MATCH_1}")
      set(PERL_VERSION_STRING "${Perl_VERSION}")
    elseif(NOT PERL_VERSION_RESULT_VARIABLE AND PERL_VERSION_OUTPUT_VARIABLE MATCHES "This is perl, version ([0-9\\._]+) +")
      set(Perl_VERSION "${CMAKE_MATCH_1}")
      set(PERL_VERSION_STRING "${Perl_VERSION}")
    endif()
  endif()
endif()

# Deprecated settings for compatibility with CMake1.4
set(PERL ${PERL_EXECUTABLE})

include(FindPackageHandleStandardArgs)
if (CMAKE_FIND_PACKAGE_NAME STREQUAL "PerlLibs")
  # FindPerlLibs include()'s this module. It's an old pattern, but rather than
  # trying to suppress this from outside the module (which is then sensitive to
  # the contents, detect the case in this module and suppress it explicitly.
  set(FPHSA_NAME_MISMATCHED 1)
endif ()
find_package_handle_standard_args(Perl
                                  REQUIRED_VARS PERL_EXECUTABLE
                                  VERSION_VAR Perl_VERSION)
unset(FPHSA_NAME_MISMATCHED)

mark_as_advanced(PERL_EXECUTABLE)
