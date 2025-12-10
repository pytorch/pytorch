# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindPerlLibs
------------

Finds Perl libraries:

.. code-block:: cmake

  find_package(PerlLibs [<version>] [...])

Perl is a general-purpose, interpreted, dynamic programming language.

This module detects whether Perl interpreter is installed via the
:module:`FindPerl` module and determines the locations of Perl include paths,
libraries, and the library name.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``PerlLibs_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether (the requested version of) ``<perl.h>`` and
  ``libperl`` were found.

``PerlLibs_VERSION``
  .. versionadded:: 4.2

  The version of Perl library found.

``PERL_SITESEARCH``
  Path to the sitesearch install directory (``-V:installsitesearch``).
``PERL_SITEARCH``
  Path to the sitelib install directory (``-V:installsitearch``).
``PERL_SITELIB``
  Path to the sitelib install directory (``-V:installsitelib``).
``PERL_VENDORARCH``
  Path to the vendor arch install directory (``-V:installvendorarch``).
``PERL_VENDORLIB``
  Path to the vendor lib install directory (``-V:installvendorlib``).
``PERL_ARCHLIB``
  Path to the core arch lib install directory (``-V:archlib``).
``PERL_PRIVLIB``
  Path to the core priv lib install directory (``-V:privlib``).
``PERL_UPDATE_ARCHLIB``
  Path to the update arch lib install directory (``-V:installarchlib``).
``PERL_UPDATE_PRIVLIB``
  Path to the update priv lib install directory (``-V:installprivlib``).
``PERL_EXTRA_C_FLAGS``
  Compilation flags used to build Perl.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``PERL_INCLUDE_PATH``
  Directory containing ``perl.h`` and other Perl header files.
``PERL_LIBRARY``
  Path to the ``libperl``.
``PERL_EXECUTABLE``
  Full path to the ``perl`` executable.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``PERLLIBS_FOUND``
  .. deprecated:: 4.2
    Use ``PerlLibs_FOUND``, which has the same value.

  Boolean indicating whether (the requested version of) ``<perl.h>`` and
  ``libperl`` were found.

Examples
^^^^^^^^

Finding Perl libraries and specifying the minimum required version:

.. code-block:: cmake

  find_package(PerlLibs 6.0)

See Also
^^^^^^^^

* The :module:`FindPerl` module to find the Perl interpreter.
#]=======================================================================]

# find the perl executable
include(${CMAKE_CURRENT_LIST_DIR}/FindPerl.cmake)

set(PerlLibs_VERSION "${Perl_VERSION}")

if (PERL_EXECUTABLE)

  function (perl_get_info _pgi_info tag)
    cmake_parse_arguments(_PGI "IS_PATH" "" "" ${ARGN})

    set (${_pgi_info} NOTFOUND PARENT_SCOPE)

    execute_process(COMMAND "${PERL_EXECUTABLE}" -V:${tag}
      OUTPUT_VARIABLE result
      RESULT_VARIABLE status)

    if (NOT status)
      string(REGEX REPLACE "${tag}='([^']*)'.*" "\\1" result "${result}")
      if (_PGI_IS_PATH)
        file(TO_CMAKE_PATH "${result}" result)
      endif()
      set (${_pgi_info} "${result}" PARENT_SCOPE)
    endif ()
  endfunction()

  ### PERL_PREFIX
  perl_get_info(PERL_PREFIX prefix IS_PATH)

  ### PERL_ARCHNAME
  perl_get_info(PERL_ARCHNAME archname)

  ### PERL_EXTRA_C_FLAGS
  perl_get_info(PERL_EXTRA_C_FLAGS cppflags)

  ### PERL_SITESEARCH
  perl_get_info(PERL_SITESEARCH installsitesearch IS_PATH)

  ### PERL_SITEARCH
  perl_get_info(PERL_SITEARCH installsitearch IS_PATH)

  ### PERL_SITELIB
  perl_get_info(PERL_SITELIB installsitelib IS_PATH)

  ### PERL_VENDORARCH
  perl_get_info(PERL_VENDORARCH installvendorarch IS_PATH)

  ### PERL_VENDORLIB
  perl_get_info(PERL_VENDORLIB installvendorlib IS_PATH)

  ### PERL_ARCHLIB
  perl_get_info(PERL_ARCHLIB archlib IS_PATH)

  ### PERL_PRIVLIB
  perl_get_info(PERL_PRIVLIB privlib IS_PATH)

  ### PERL_UPDATE_ARCHLIB
  perl_get_info(PERL_UPDATE_ARCHLIB installarchlib IS_PATH)

  ### PERL_UPDATE_PRIVLIB
  perl_get_info(PERL_UPDATE_PRIVLIB installprivlib IS_PATH)

  ### PERL_POSSIBLE_LIBRARY_NAMES
  perl_get_info(PERL_POSSIBLE_LIBRARY_NAMES libperl)
  if (NOT PERL_POSSIBLE_LIBRARY_NAMES)
    set(PERL_POSSIBLE_LIBRARY_NAMES perl${PerlLibs_VERSION} perl)
  endif()
  if(CMAKE_SYSTEM_NAME MATCHES "CYGWIN")
    list (APPEND PERL_POSSIBLE_LIBRARY_NAMES perl${PerlLibs_VERSION})
  endif()
  if (CMAKE_SYSTEM_NAME MATCHES "MSYS|CYGWIN")
    # On MSYS and CYGWIN environments, current perl -V:libperl gives shared
    # library name rather than the import library. So, extend possible library
    # names.
    list (APPEND PERL_POSSIBLE_LIBRARY_NAMES perl)
  endif()

  ### PERL_INCLUDE_PATH
  find_path(PERL_INCLUDE_PATH
    NAMES
      perl.h
    PATHS
      "${PERL_UPDATE_ARCHLIB}/CORE"
      "${PERL_ARCHLIB}/CORE"
      /usr/lib/perl5/${PerlLibs_VERSION}/${PERL_ARCHNAME}/CORE
      /usr/lib/perl/${PerlLibs_VERSION}/${PERL_ARCHNAME}/CORE
      /usr/lib/perl5/${PerlLibs_VERSION}/CORE
      /usr/lib/perl/${PerlLibs_VERSION}/CORE
  )

  ### PERL_LIBRARY
  find_library(PERL_LIBRARY
    NAMES
      ${PERL_POSSIBLE_LIBRARY_NAMES}
    PATHS
      "${PERL_UPDATE_ARCHLIB}/CORE"
      "${PERL_ARCHLIB}/CORE"
      /usr/lib/perl5/${PerlLibs_VERSION}/${PERL_ARCHNAME}/CORE
      /usr/lib/perl/${PerlLibs_VERSION}/${PERL_ARCHNAME}/CORE
      /usr/lib/perl5/${PerlLibs_VERSION}/CORE
      /usr/lib/perl/${PerlLibs_VERSION}/CORE
  )

endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PerlLibs REQUIRED_VARS PERL_LIBRARY PERL_INCLUDE_PATH
                                           VERSION_VAR PerlLibs_VERSION)

# Introduced after CMake 2.6.4 to bring module into compliance
set(PERL_INCLUDE_DIR  ${PERL_INCLUDE_PATH})
set(PERL_INCLUDE_DIRS ${PERL_INCLUDE_PATH})
set(PERL_LIBRARIES    ${PERL_LIBRARY})
# For backward compatibility with CMake before 2.8.8
set(PERL_VERSION ${PerlLibs_VERSION})

mark_as_advanced(
  PERL_INCLUDE_PATH
  PERL_LIBRARY
)
