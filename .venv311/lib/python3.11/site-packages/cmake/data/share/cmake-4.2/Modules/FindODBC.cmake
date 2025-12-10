# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindODBC
--------

.. versionadded:: 3.12

Finds the Open Database Connectivity (ODBC) library, which implements a
standard API for accessing database systems:

.. code-block:: cmake

  find_package(ODBC [...])

ODBC enables applications to communicate with different database management
systems (DBMS) using a common set of functions.  Communication with a specific
database is handled through ODBC drivers, which the library loads at runtime.

On Windows, when building with Visual Studio, this module assumes the ODBC
library is provided by the available Windows SDK.

On Unix-like systems, this module searches for ODBC library provided by unixODBC
or iODBC implementations of ODBC API.  By default, this module looks for the
ODBC config program to determine the ODBC library and include directory, first
from unixODBC, then from iODBC.  If no config program is found, it searches for
ODBC header and library in standard locations.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``ODBC::ODBC``
  Target encapsulating the ODBC usage requirements, available if ODBC is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``ODBC_FOUND``
  Boolean indicating whether ODBC was found.

``ODBC_INCLUDE_DIRS``
  Include directories containing headers needed to use ODBC.

``ODBC_LIBRARIES``
  Libraries needed to link against to use ODBC.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``ODBC_INCLUDE_DIR``
  The path to the directory containing ``sql.h`` and other ODBC headers.  May be
  empty on Windows, where the include directory corresponding to the expected
  Windows SDK is already available in the compilation environment.

``ODBC_LIBRARY``
  The path to the ODBC library or a library name.  On Windows, this may be only
  a library name, because the library directory corresponding to the expected
  Windows SDK is already available in the compilation environment.

``ODBC_CONFIG``
  The path to the ODBC config program if found or specified.  For example,
  ``odbc_config`` for unixODBC, or ``iodbc-config`` for iODBC.

Limitations
^^^^^^^^^^^

* On Windows, this module does not search for iODBC.
* On Unix-like systems, there is no built-in mechanism to prefer unixODBC over
  iODBC, or vice versa.  To bypass this limitation, explicitly set the
  ``ODBC_CONFIG`` variable to the path of the desired ODBC config program.
* This module does not support searching for or selecting a specific ODBC
  driver.

Examples
^^^^^^^^

Example: Finding and Using ODBC
"""""""""""""""""""""""""""""""

Finding ODBC and linking it to a project target:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  find_package(ODBC)
  target_link_libraries(project_target PRIVATE ODBC::ODBC)

Example: Finding a Custom ODBC Installation
"""""""""""""""""""""""""""""""""""""""""""

The following examples are for Unix-like systems and demonstrate how to set hint
and cache variables during the CMake configuration phase to help this module
find a custom ODBC implementation (e.g. one not supported by default).

To specify the installation prefix using :variable:`CMAKE_PREFIX_PATH`:

.. code-block:: console

  $ cmake -D CMAKE_PREFIX_PATH=/path/to/odbc-installation -B build

Or using the dedicated :variable:`ODBC_ROOT <<PackageName>_ROOT>` variable:

.. code-block:: console

  $ cmake -D ODBC_ROOT=/path/to/odbc-installation -B build

To manually specify the ODBC config program, if available, so that the ODBC
installation can be automatically determined based on the config tool:

.. code-block:: console

  $ cmake -D ODBC_CONFIG=/path/to/odbc/bin/odbc-config -B build

To manually specify the ODBC library and include directory:

.. code-block:: console

  $ cmake \
      -D ODBC_LIBRARY=/path/to/odbc/lib/libodbc.so \
      -D ODBC_INCLUDE_DIR=/path/to/odbc/include \
      -B build
#]=======================================================================]

# Define internal variables
set(_odbc_include_paths)
set(_odbc_lib_paths)
set(_odbc_lib_names)
set(_odbc_required_libs_names)

### Try Windows Kits ##########################################################
if(WIN32)
  # List names of ODBC libraries on Windows
  if(NOT MINGW)
    set(ODBC_LIBRARY odbc32.lib)
  else()
    set(ODBC_LIBRARY libodbc32.a)
  endif()
  set(_odbc_lib_names odbc32;)

  # List additional libraries required to use ODBC library
  if(MSVC OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(_odbc_required_libs_names odbccp32;ws2_32)
  elseif(MINGW)
    set(_odbc_required_libs_names odbccp32)
  endif()
endif()

### Try unixODBC or iODBC config program ######################################
if (UNIX)
  find_program(ODBC_CONFIG
    NAMES odbc_config iodbc-config
    DOC "Path to unixODBC or iODBC config program")
  mark_as_advanced(ODBC_CONFIG)
endif()

if (UNIX AND ODBC_CONFIG)
  # unixODBC and iODBC accept unified command line options
  execute_process(COMMAND ${ODBC_CONFIG} --cflags
    OUTPUT_VARIABLE _cflags OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${ODBC_CONFIG} --libs
    OUTPUT_VARIABLE _libs OUTPUT_STRIP_TRAILING_WHITESPACE)

  # Collect paths of include directories from CFLAGS
  separate_arguments(_cflags NATIVE_COMMAND "${_cflags}")
  foreach(arg IN LISTS _cflags)
    if("${arg}" MATCHES "^-I(.*)$")
      list(APPEND _odbc_include_paths "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  unset(_cflags)

  # Collect paths of library names and directories from LIBS
  separate_arguments(_libs NATIVE_COMMAND "${_libs}")
  foreach(arg IN LISTS _libs)
    if("${arg}" MATCHES "^-L(.*)$")
      list(APPEND _odbc_lib_paths "${CMAKE_MATCH_1}")
    elseif("${arg}" MATCHES "^-l(.*)$")
      set(_lib_name ${CMAKE_MATCH_1})
      string(REGEX MATCH "odbc" _is_odbc ${_lib_name})
      if(_is_odbc)
        list(APPEND _odbc_lib_names ${_lib_name})
      else()
        list(APPEND _odbc_required_libs_names ${_lib_name})
      endif()
      unset(_lib_name)
    endif()
  endforeach()
  unset(_libs)
endif()

### Try unixODBC or iODBC in include/lib filesystems ##########################
if (UNIX AND NOT ODBC_CONFIG)
  # List names of both ODBC libraries, unixODBC and iODBC
  set(_odbc_lib_names odbc;iodbc;unixodbc;)
endif()

### Find include directories ##################################################
find_path(ODBC_INCLUDE_DIR
  NAMES sql.h
  PATHS ${_odbc_include_paths})

if(NOT ODBC_INCLUDE_DIR AND WIN32)
  set(ODBC_INCLUDE_DIR "")
endif()

### Find libraries ############################################################
if(NOT ODBC_LIBRARY)
  find_library(ODBC_LIBRARY
    NAMES ${_odbc_lib_names}
    PATHS ${_odbc_lib_paths}
    PATH_SUFFIXES odbc)

  foreach(_lib IN LISTS _odbc_required_libs_names)
    find_library(_lib_path
      NAMES ${_lib}
      PATHS ${_odbc_lib_paths} # system paths or collected from ODBC_CONFIG
      PATH_SUFFIXES odbc)
    if(_lib_path)
      list(APPEND _odbc_required_libs_paths ${_lib_path})
    endif()
    unset(_lib_path CACHE)
  endforeach()
endif()

# Unset internal variables as no longer used
unset(_odbc_include_paths)
unset(_odbc_lib_paths)
unset(_odbc_lib_names)
unset(_odbc_required_libs_names)

### Set result variables ######################################################
set(_odbc_required_vars ODBC_LIBRARY)
if(NOT WIN32)
  list(APPEND _odbc_required_vars ODBC_INCLUDE_DIR)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ODBC DEFAULT_MSG ${_odbc_required_vars})

unset(_odbc_required_vars)

mark_as_advanced(ODBC_LIBRARY ODBC_INCLUDE_DIR)

set(ODBC_INCLUDE_DIRS ${ODBC_INCLUDE_DIR})
list(APPEND ODBC_LIBRARIES ${ODBC_LIBRARY})
list(APPEND ODBC_LIBRARIES ${_odbc_required_libs_paths})

### Import targets ############################################################
if(ODBC_FOUND)
  if(NOT TARGET ODBC::ODBC)
    if(IS_ABSOLUTE "${ODBC_LIBRARY}")
      add_library(ODBC::ODBC UNKNOWN IMPORTED)
      set_target_properties(ODBC::ODBC PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${ODBC_LIBRARY}")
    else()
      add_library(ODBC::ODBC INTERFACE IMPORTED)
      set_target_properties(ODBC::ODBC PROPERTIES
        IMPORTED_LIBNAME "${ODBC_LIBRARY}")
    endif()
    set_target_properties(ODBC::ODBC PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${ODBC_INCLUDE_DIR}")

    if(_odbc_required_libs_paths)
      set_property(TARGET ODBC::ODBC APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES "${_odbc_required_libs_paths}")
    endif()
  endif()
endif()

unset(_odbc_required_libs_paths)
