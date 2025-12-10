# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindIce
-------

.. versionadded:: 3.1

Finds the Internet Communication Engine (Ice) programs, libraries and datafiles:

.. code-block:: cmake

  find_package(Ice [<version>] COMPONENTS <components>... [...])

Ice is an open-source remote procedure call (RPC) framework developed by ZeroC
and provides SDKs for various languages to develop network applications.

.. versionadded:: 3.4
  Imported targets for components and many new ``*_EXECUTABLE`` variables.

.. versionadded:: 3.7
  Debug and Release library variants are found separately.

.. versionadded:: 3.10
  Ice 3.7 support, including new components, programs and the Nuget package.

Components
^^^^^^^^^^

Ice consists of several libraries and programs (executables).  This find module
supports components, which can be specified using the :command:`find_package`
command, to select specific Ice libraries for use in a CMake project.
Executables provided by Ice installation are always searched automatically,
regardless of the specified components.

The list of available components depends on the Ice version in use.  To
successfully find Ice, at least one component must be specified:

.. code-block:: cmake

  find_package(Ice COMPONENTS <components>...)

Supported components include:

.. hlist::

  - ``Freeze``
  - ``Glacier2``
  - ``Ice``
  - ``IceBox``
  - ``IceDB``
  - ``IceDiscovery``
  - ``IceGrid``
  - ``IceLocatorDiscovery``
  - ``IcePatch``
  - ``IceSSL``
  - ``IceStorm``
  - ``IceUtil``
  - ``IceXML``
  - ``Slice``

Ice 3.7 and later also include C++11-specific components:

.. hlist::

  - ``Glacier2++11``
  - ``Ice++11``
  - ``IceBox++11``
  - ``IceDiscovery++11``
  - ``IceGrid++11``
  - ``IceLocatorDiscovery++11``
  - ``IceSSL++11``
  - ``IceStorm++11``

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``Ice::<component>``
  .. versionadded:: 3.4

  Target encapsulating the usage requirements for the specified Ice component
  (library), available if that component is found.  The ``<component>`` should
  be written in the same case, as listed above.  For example, use
  ``Ice::Glacier2`` for the Ice Glacier2 library, or ``Ice::Ice++11`` for the
  Ice++11 library, etc.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Ice_FOUND``
  Boolean indicating whether the (requested version of) main programs,
  libraries and all requested components for using Ice were found.

``Ice_VERSION``
  The version of Ice release found.

``Ice_INCLUDE_DIRS``
  The include directories containing headers needed to use Ice.

``Ice_LIBRARIES``
  Component libraries needed to link against to use Ice.

``Ice_SLICE_DIRS``
  The data directories containing interface definitions (``*.ice`` files) for
  Slice (Specification Language for Ice).

Ice component libraries are stored in:

``Ice_<COMPONENT>_FOUND``
  Boolean indicating whether the specified Ice component was found.  The
  ``<COMPONENT>`` should be written in uppercase.

``Ice_<COMPONENT>_LIBRARIES``
  Libraries provided by the specified Ice component.  The ``<COMPONENT>`` should
  be written in uppercase.

Slice programs are stored in:

``Ice_SLICE2CONFLUENCE_EXECUTABLE``
  .. versionadded:: 3.14

  The path to the ``slice2confluence`` executable.

``Ice_SLICE2CPP_EXECUTABLE``
  The path to the ``slice2cpp`` executable.

``Ice_SLICE2CS_EXECUTABLE``
  The path to the ``slice2cs`` executable.

``Ice_SLICE2FREEZEJ_EXECUTABLE``
  The path to the ``slice2freezej`` executable.

``Ice_SLICE2FREEZE_EXECUTABLE``
  The path to the ``slice2freeze`` executable.

``Ice_SLICE2HTML_EXECUTABLE``
  The path to the ``slice2html`` executable.

``Ice_SLICE2JAVA_EXECUTABLE``
  The path to the ``slice2java`` executable.

``Ice_SLICE2JS_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``slice2js`` executable.

``Ice_SLICE2MATLAB_EXECUTABLE``
  .. versionadded:: 3.14

  The path to the ``slice2matlab`` executable.

``Ice_SLICE2OBJC_EXECUTABLE``
  .. versionadded:: 3.10

  The path to the ``slice2objc`` executable.

``Ice_SLICE2PHP_EXECUTABLE``
  The path to the ``slice2php`` executable.

``Ice_SLICE2PY_EXECUTABLE``
  The path to the ``slice2py`` executable.

``Ice_SLICE2RB_EXECUTABLE``
  The path to the ``slice2rb`` executable.

Ice programs are stored in:

``Ice_GLACIER2ROUTER_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``glacier2router`` executable.

``Ice_ICEBOX_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``icebox`` executable.

``Ice_ICEBOX++11_EXECUTABLE``
  .. versionadded:: 3.10

  The path to the ``icebox++11`` executable.

``Ice_ICEBOXADMIN_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``iceboxadmin`` executable.

``Ice_ICEBOXD_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``iceboxd`` executable.

``Ice_ICEBOXNET_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``iceboxnet`` executable.

``Ice_ICEBRIDGE_EXECUTABLE``
  .. versionadded:: 3.10

  The path to the ``icebridge`` executable.

``Ice_ICEGRIDADMIN_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``icegridadmin`` executable.

``Ice_ICEGRIDDB_EXECUTABLE``
  .. versionadded:: 3.10

  The path to the ``icegriddb`` executable.

``Ice_ICEGRIDNODE_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``icegridnode`` executable.

``Ice_ICEGRIDNODED_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``icegridnoded`` executable.

``Ice_ICEGRIDREGISTRY_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``icegridregistry`` executable.

``Ice_ICEGRIDREGISTRYD_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``icegridregistryd`` executable.

``Ice_ICEPATCH2CALC_EXECUTABLE``
  The path to the ``icepatch2calc`` executable.

``Ice_ICEPATCH2CLIENT_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``icepatch2client`` executable.

``Ice_ICEPATCH2SERVER_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``icepatch2server`` executable.

``Ice_ICESERVICEINSTALL_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``iceserviceinstall`` executable.

``Ice_ICESTORMADMIN_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``icestormadmin`` executable.

``Ice_ICESTORMDB_EXECUTABLE``
  .. versionadded:: 3.10

  The path to the ``icestormdb`` executable.

``Ice_ICESTORMMIGRATE_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``icestormmigrate`` executable.

Ice database programs are stored in the following variables (on Windows, they
are included with the Ice installation; on other platforms, they are usually
available through standard Berkeley DB packages):

``Ice_DB_ARCHIVE_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``db_archive`` executable.

``Ice_DB_CHECKPOINT_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``db_checkpoint`` executable.

``Ice_DB_DEADLOCK_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``db_deadlock`` executable.

``Ice_DB_DUMP_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``db_dump`` executable.

``Ice_DB_HOTBACKUP_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``db_hotbackup`` executable.

``Ice_DB_LOAD_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``db_load`` executable.

``Ice_DB_LOG_VERIFY_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``db_log_verify`` executable.

``Ice_DB_PRINTLOG_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``db_printlog`` executable.

``Ice_DB_RECOVER_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``db_recover`` executable.

``Ice_DB_STAT_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``db_stat`` executable.

``Ice_DB_TUNER_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``db_tuner`` executable.

``Ice_DB_UPGRADE_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``db_upgrade`` executable.

``Ice_DB_VERIFY_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``db_verify`` executable.

``Ice_DUMPDB_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``dumpdb`` executable.

``Ice_TRANSFORMDB_EXECUTABLE``
  .. versionadded:: 3.4

  The path to the ``transformdb`` executable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Ice_<PROGRAM>_EXECUTABLE``
  The path to the specified ``<PROGRAM>`` executable;  The ``<PROGRAM>`` is the
  uppercase name of the Ice program as listed in above result variables of
  executables.

``Ice_INCLUDE_DIR``
  The directory containing Ice headers.

``Ice_SLICE_DIR``
  The data directory containing interface definitions for Slice.

``Ice_<COMPONENT>_LIBRARY``
  The path to the library for the specified component.  The ``<COMPONENT>``
  should be written in uppercase.

Hints
^^^^^

This module accepts the following variables:

``Ice_HOME``
  Set this CMake variable to the root of the Ice installation in order to
  search for Ice in a custom location.

  .. note::

    On Windows, Ice 3.7.0 and later provide libraries via the NuGet
    package manager.  Appropriate NuGet packages will be searched for
    using :variable:`CMAKE_PREFIX_PATH`, or alternatively ``Ice_HOME`` may be
    set to the location of a specific NuGet package to restrict the search.

``ICE_HOME``
  Environment variable (uppercased) may also be set to the root of the Ice
  installation;  The ``Ice_HOME`` CMake variable takes precedence.

``Ice_DEBUG``
  Set this variable to boolean true to enable debug output from this module.

.. note::

  In most cases, none of the above variables need to be set unless multiple Ice
  versions are installed and a specific one is required.  On Windows, the most
  recent version is typically found using the registry.  On Unix-like systems,
  programs, headers, and libraries are usually found in standard locations,
  although ``Ice_SLICE_DIRS`` might not be detected automatically (commonly
  known locations are searched).  All other variables default based on the value
  of ``Ice_HOME``, if set.

  It's also possible to set ``Ice_HOME`` while selectively overriding specific
  locations for individual components;  This might be required, for example, in
  newer versions of Visual Studio if the heuristics are not sufficient to
  identify the correct programs and libraries for the specific Visual Studio
  version.

Examples
^^^^^^^^

Finding the Ice core library and linking it to a project target:

.. code-block:: cmake

  find_package(Ice COMPONENTS Ice)
  target_link_libraries(project_target PRIVATE Ice::Ice)

Finding Ice core library and IceSSL library, and linking them to a project
target:

.. code-block:: cmake

  find_package(Ice COMPONENTS Ice IceSSL)
  target_link_libraries(project_target PRIVATE Ice::Ice Ice::IceSSL)

Finding Ice core library as required component and Ice Freeze library as
optional:

.. code-block:: cmake

  find_package(Ice COMPONENTS Ice OPTIONAL_COMPONENTS Freeze)
#]=======================================================================]

# Written by Roger Leigh <rleigh@codelibre.net>

  set(_Ice_db_programs
      db_archive
      db_checkpoint
      db_deadlock
      db_dump
      db_hotbackup
      db_load
      db_log_verify
      db_printlog
      db_recover
      db_stat
      db_tuner
      db_upgrade
      db_verify
      dumpdb
      transformdb)

  set(_Ice_programs
      glacier2router
      icebox
      icebox++11
      iceboxadmin
      iceboxd
      iceboxnet
      icebridge
      icegridadmin
      icegriddb
      icegridnode
      icegridnoded
      icegridregistry
      icegridregistryd
      icepatch2calc
      icepatch2client
      icepatch2server
      iceserviceinstall
      icestormadmin
      icestormdb
      icestormmigrate)

  set(_Ice_slice_programs
      slice2confluence
      slice2cpp
      slice2cs
      slice2freezej
      slice2freeze
      slice2html
      slice2java
      slice2js
      slice2matlab
      slice2objc
      slice2php
      slice2py
      slice2rb)


# The Ice checks are contained in a function due to the large number
# of temporary variables needed.
function(_Ice_FIND)
  # Released versions of Ice, including generic short forms
  set(ice_versions
      3
      3.7
      3.7.0
      3.6
      3.6.3
      3.6.2
      3.6.1
      3.6.0
      3.5
      3.5.1
      3.5.0
      3.4
      3.4.2
      3.4.1
      3.4.0
      3.3
      3.3.1
      3.3.0)

  foreach(ver ${ice_versions})
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\$" two_digit_version_match "${ver}")
    if(two_digit_version_match)
      string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\$" "\\1\\2" two_digit_version "${ver}")
      list(APPEND ice_suffix_versions "${two_digit_version}")
    endif()
  endforeach()

  # Set up search paths, taking compiler into account.  Search Ice_HOME,
  # with ICE_HOME in the environment as a fallback if unset.
  if(Ice_HOME)
    list(APPEND ice_roots "${Ice_HOME}")
  else()
    if(NOT "$ENV{ICE_HOME}" STREQUAL "")
      file(TO_CMAKE_PATH "$ENV{ICE_HOME}" NATIVE_PATH)
      list(APPEND ice_roots "${NATIVE_PATH}")
      set(Ice_HOME "${NATIVE_PATH}"
          CACHE PATH "Location of the Ice installation" FORCE)
    endif()
  endif()

  set(_bin "bin/Win32")
  set(_lib "lib/Win32")
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_bin "bin/x64")
    set(_lib "lib/x64")
    # 64-bit path suffix
    set(_x64 "/x64")
    # 64-bit library directory
    set(_lib64 "lib64")
  endif()

  unset(vcvers)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" OR "${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC")
    if(MSVC_TOOLSET_VERSION GREATER_EQUAL 141)
      set(vcvers "141;140")
    elseif(MSVC_TOOLSET_VERSION GREATER_EQUAL 100)
      set(vcvers "${MSVC_TOOLSET_VERSION}")
    elseif(MSVC_TOOLSET_VERSION GREATER_EQUAL 90)
      set(vcvers "${MSVC_TOOLSET_VERSION}")
      set(vcyear "2008")
    elseif(MSVC_TOOLSET_VERSION GREATER_EQUAL 80)
      set(vcvers "${MSVC_TOOLSET_VERSION}")
      set(vcyear "2005")
    else() # Unknown version
      set(vcvers Unknown)
    endif()
  endif()

  # For compatibility with ZeroC Windows builds.
  if(vcvers)
    list(APPEND ice_binary_suffixes "build/native/${_bin}/Release" "tools")
    list(APPEND ice_debug_library_suffixes "build/native/${_lib}/Debug")
    list(APPEND ice_release_library_suffixes "build/native/${_lib}/Release")
    foreach(vcver IN LISTS vcvers)
      # Earlier Ice (3.3) builds don't use vcnnn subdirectories, but are harmless to check.
      list(APPEND ice_binary_suffixes "bin/vc${vcver}${_x64}" "bin/vc${vcver}")
      list(APPEND ice_debug_library_suffixes "lib/vc${vcver}${_x64}" "lib/vc${vcver}")
      list(APPEND ice_release_library_suffixes "lib/vc${vcver}${_x64}" "lib/vc${vcver}")
    endforeach()
  endif()
  # Generic 64-bit and 32-bit directories
  list(APPEND ice_binary_suffixes "bin${_x64}" "bin")
  list(APPEND ice_debug_library_suffixes "libx32" "${_lib64}" "lib${_x64}" "lib")
  list(APPEND ice_release_library_suffixes "libx32" "${_lib64}" "lib${_x64}" "lib")
  if(vcvers)
    list(APPEND ice_include_suffixes "build/native/include")
  endif()
  list(APPEND ice_include_suffixes "include")
  list(APPEND ice_slice_suffixes "slice")

  # On Windows, look in the registry for install locations.  Different
  # versions of Ice install support different compiler versions.
  if(vcvers)
    foreach(ice_version ${ice_versions})
      foreach(vcver IN LISTS vcvers)
        list(APPEND ice_nuget_dirs "zeroc.ice.v${vcver}.${ice_version}")
        list(APPEND freeze_nuget_dirs "zeroc.freeze.v${vcver}.${ice_version}")
      endforeach()
    endforeach()
    find_path(Ice_NUGET_DIR
              NAMES "tools/slice2cpp.exe"
              PATH_SUFFIXES ${ice_nuget_dirs}
              DOC "Ice NuGet directory")
    if(Ice_NUGET_DIR)
      list(APPEND ice_roots "${Ice_NUGET_DIR}")
    endif()
    find_path(Freeze_NUGET_DIR
              NAMES "tools/slice2freeze.exe"
              PATH_SUFFIXES ${freeze_nuget_dirs}
              DOC "Freeze NuGet directory")
    if(Freeze_NUGET_DIR)
      list(APPEND ice_roots "${Freeze_NUGET_DIR}")
    endif()
    foreach(ice_version ${ice_versions})
      # Ice 3.3 releases use a Visual Studio year suffix and value is
      # enclosed in double quotes, though only the leading quote is
      # returned by get_filename_component.
      unset(ice_location)
      if(vcyear)
        get_filename_component(ice_location
                               "[HKEY_LOCAL_MACHINE\\SOFTWARE\\ZeroC\\Ice ${ice_version} for Visual Studio ${vcyear};InstallDir]"
                               PATH)
        if(ice_location AND NOT ("${ice_location}" STREQUAL "/registry" OR "${ice_location}" STREQUAL "/"))
          string(REGEX REPLACE "^\"(.*)\"?$" "\\1" ice_location "${ice_location}")
          get_filename_component(ice_location "${ice_location}" ABSOLUTE)
        else()
          unset(ice_location)
        endif()
      endif()
      # Ice 3.4+ releases don't use a suffix
      if(NOT ice_location OR "${ice_location}" STREQUAL "/registry")
        get_filename_component(ice_location
                               "[HKEY_LOCAL_MACHINE\\SOFTWARE\\ZeroC\\Ice ${ice_version};InstallDir]"
                               ABSOLUTE)
      endif()

      if(ice_location AND NOT "${ice_location}" STREQUAL "/registry")
        list(APPEND ice_roots "${ice_location}")
      endif()
    endforeach()
  else()
    foreach(ice_version ${ice_versions})
      # Prefer 64-bit variants if present (and using a 64-bit compiler)
      list(APPEND ice_roots "/opt/Ice-${ice_version}")
    endforeach()
  endif()

  # Find all Ice programs
  foreach(program ${_Ice_db_programs} ${_Ice_programs} ${_Ice_slice_programs})
    string(TOUPPER "${program}" program_upcase)
    set(cache_var "Ice_${program_upcase}_EXECUTABLE")
    set(program_var "Ice_${program_upcase}_EXECUTABLE")
    find_program("${cache_var}" "${program}"
      HINTS ${ice_roots}
      PATH_SUFFIXES ${ice_binary_suffixes}
      DOC "Ice ${program} executable")
    mark_as_advanced(cache_var)
    set("${program_var}" "${${cache_var}}" PARENT_SCOPE)
  endforeach()

  # Get version.
  if(Ice_SLICE2CPP_EXECUTABLE)
    # Execute in C locale for safety
    set(_Ice_SAVED_LC_ALL "$ENV{LC_ALL}")
    set(ENV{LC_ALL} C)

    execute_process(COMMAND ${Ice_SLICE2CPP_EXECUTABLE} --version
      ERROR_VARIABLE Ice_VERSION_SLICE2CPP_FULL
      ERROR_STRIP_TRAILING_WHITESPACE)

    # restore the previous LC_ALL
    set(ENV{LC_ALL} ${_Ice_SAVED_LC_ALL})

    # Make short version
    string(REGEX REPLACE "^(.*)\\.[^.]*$" "\\1" Ice_VERSION_SLICE2CPP_SHORT "${Ice_VERSION_SLICE2CPP_FULL}")
    set(Ice_VERSION "${Ice_VERSION_SLICE2CPP_FULL}" PARENT_SCOPE)
  endif()

  if(NOT Ice_FIND_QUIETLY)
    message(STATUS "Ice version: ${Ice_VERSION_SLICE2CPP_FULL}")
  endif()

  # Find include directory
  find_path(Ice_INCLUDE_DIR
            NAMES "Ice/Ice.h"
            HINTS ${ice_roots}
            PATH_SUFFIXES ${ice_include_suffixes}
            DOC "Ice include directory")
  set(Ice_INCLUDE_DIR "${Ice_INCLUDE_DIR}" PARENT_SCOPE)

  find_path(Freeze_INCLUDE_DIR
            NAMES "Freeze/Freeze.h"
            HINTS ${ice_roots}
            PATH_SUFFIXES ${ice_include_suffixes}
            DOC "Freeze include directory")
  set(Freeze_INCLUDE_DIR "${Freeze_INCLUDE_DIR}" PARENT_SCOPE)

  # In common use on Linux, MacOS X (homebrew) and FreeBSD; prefer
  # version-specific dir
  list(APPEND ice_slice_paths
       /usr/local/share /usr/share)
  list(APPEND ice_slice_suffixes
       "Ice-${Ice_VERSION_SLICE2CPP_FULL}/slice"
       "Ice-${Ice_VERSION_SLICE2CPP_SHORT}/slice"
       "ice/slice"
       Ice)

  # Find slice directory
  find_path(Ice_SLICE_DIR
            NAMES "Ice/Connection.ice"
            HINTS ${ice_roots}
                  ${ice_slice_paths}
            PATH_SUFFIXES ${ice_slice_suffixes}
            NO_DEFAULT_PATH
            DOC "Ice slice directory")
  set(Ice_SLICE_DIR "${Ice_SLICE_DIR}" PARENT_SCOPE)

  # Find all Ice libraries
  set(Ice_REQUIRED_LIBS_FOUND ON)
  foreach(component ${Ice_FIND_COMPONENTS})
    string(TOUPPER "${component}" component_upcase)
    set(component_cache "Ice_${component_upcase}_LIBRARY")
    set(component_cache_release "${component_cache}_RELEASE")
    set(component_cache_debug "${component_cache}_DEBUG")
    set(component_found "${component_upcase}_FOUND")
    set(component_library "${component}")
    unset(component_library_release_names)
    unset(component_library_debug_names)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" OR "${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC")
      string(REGEX MATCH ".+\\+\\+11$" component_library_cpp11 "${component_library}")
      if(component_library_cpp11)
        string(REGEX REPLACE "^(.+)(\\+\\+11)$" "\\1" component_library "${component_library}")
      endif()
      foreach(suffix_ver ${ice_suffix_versions})
        set(_name "${component_library}${suffix_ver}")
        if(component_library_cpp11)
          string(APPEND _name "++11")
        endif()
        list(APPEND component_library_debug_names "${_name}d")
        list(APPEND component_library_release_names "${_name}")
      endforeach()
      set(_name "${component_library}")
      if(component_library_cpp11)
        string(APPEND _name "++11")
      endif()
      list(APPEND component_library_debug_names "${_name}d")
      list(APPEND component_library_release_names "${_name}")
    else()
      list(APPEND component_library_debug_names "${component_library}d")
      list(APPEND component_library_release_names "${component_library}")
    endif()
    find_library("${component_cache_release}" ${component_library_release_names}
      HINTS ${ice_roots}
      PATH_SUFFIXES ${ice_release_library_suffixes}
      DOC "Ice ${component} library (release)")
    find_library("${component_cache_debug}" ${component_library_debug_names}
      HINTS ${ice_roots}
      PATH_SUFFIXES ${ice_debug_library_suffixes}
      DOC "Ice ${component} library (debug)")
    include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)
    select_library_configurations(Ice_${component_upcase})
    mark_as_advanced("${component_cache_release}" "${component_cache_debug}")
    if(${component_cache})
      set("${component_found}" ON)
      list(APPEND Ice_LIBRARY "${${component_cache}}")
    endif()
    mark_as_advanced("${component_found}")
    set("${component_cache}" "${${component_cache}}" PARENT_SCOPE)
    set("${component_found}" "${${component_found}}" PARENT_SCOPE)
    if(${component_found})
      if (Ice_FIND_REQUIRED_${component})
        list(APPEND Ice_LIBS_FOUND "${component} (required)")
      else()
        list(APPEND Ice_LIBS_FOUND "${component} (optional)")
      endif()
    else()
      if (Ice_FIND_REQUIRED_${component})
        set(Ice_REQUIRED_LIBS_FOUND OFF)
        list(APPEND Ice_LIBS_NOTFOUND "${component} (required)")
      else()
        list(APPEND Ice_LIBS_NOTFOUND "${component} (optional)")
      endif()
    endif()
  endforeach()
  set(_Ice_REQUIRED_LIBS_FOUND "${Ice_REQUIRED_LIBS_FOUND}" PARENT_SCOPE)
  set(Ice_LIBRARY "${Ice_LIBRARY}" PARENT_SCOPE)

  if(NOT Ice_FIND_QUIETLY)
    if(Ice_LIBS_FOUND)
      message(STATUS "Found the following Ice libraries:")
      foreach(found ${Ice_LIBS_FOUND})
        message(STATUS "  ${found}")
      endforeach()
    endif()
    if(Ice_LIBS_NOTFOUND)
      message(STATUS "The following Ice libraries were not found:")
      foreach(notfound ${Ice_LIBS_NOTFOUND})
        message(STATUS "  ${notfound}")
      endforeach()
    endif()
  endif()

  if(Ice_DEBUG)
    message(STATUS "--------FindIce.cmake search debug--------")
    message(STATUS "ICE binary path search order: ${ice_roots}")
    message(STATUS "ICE binary suffixes: ${ice_binary_suffixes}")
    message(STATUS "ICE include path search order: ${ice_roots}")
    message(STATUS "ICE include suffixes: ${ice_include_suffixes}")
    message(STATUS "ICE slice path search order: ${ice_roots} ${ice_slice_paths}")
    message(STATUS "ICE slice suffixes: ${ice_slice_suffixes}")
    message(STATUS "ICE library path search order: ${ice_roots}")
    message(STATUS "ICE debug library suffixes: ${ice_debug_library_suffixes}")
    message(STATUS "ICE release library suffixes: ${ice_release_library_suffixes}")
    message(STATUS "----------------")
  endif()
endfunction()

_Ice_FIND()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Ice
                                  REQUIRED_VARS Ice_SLICE2CPP_EXECUTABLE
                                                Ice_INCLUDE_DIR
                                                Ice_SLICE_DIR
                                                Ice_LIBRARY
                                                _Ice_REQUIRED_LIBS_FOUND
                                  VERSION_VAR Ice_VERSION
                                  FAIL_MESSAGE "Failed to find all Ice components")

unset(_Ice_REQUIRED_LIBS_FOUND)

if(Ice_FOUND)
  set(Ice_INCLUDE_DIRS "${Ice_INCLUDE_DIR}")
  if (Freeze_INCLUDE_DIR)
    list(APPEND Ice_INCLUDE_DIRS "${Freeze_INCLUDE_DIR}")
  endif()
  set(Ice_SLICE_DIRS "${Ice_SLICE_DIR}")
  set(Ice_LIBRARIES "${Ice_LIBRARY}")
  foreach(_Ice_component ${Ice_FIND_COMPONENTS})
    string(TOUPPER "${_Ice_component}" _Ice_component_upcase)
    set(_Ice_component_cache "Ice_${_Ice_component_upcase}_LIBRARY")
    set(_Ice_component_cache_release "Ice_${_Ice_component_upcase}_LIBRARY_RELEASE")
    set(_Ice_component_cache_debug "Ice_${_Ice_component_upcase}_LIBRARY_DEBUG")
    set(_Ice_component_lib "Ice_${_Ice_component_upcase}_LIBRARIES")
    set(_Ice_component_found "${_Ice_component_upcase}_FOUND")
    set(_Ice_imported_target "Ice::${_Ice_component}")
    if(${_Ice_component_found})
      set("${_Ice_component_lib}" "${${_Ice_component_cache}}")
      if(NOT TARGET ${_Ice_imported_target})
        add_library(${_Ice_imported_target} UNKNOWN IMPORTED)
        set_target_properties(${_Ice_imported_target} PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${Ice_INCLUDE_DIRS}")
        if(EXISTS "${${_Ice_component_cache}}")
          set_target_properties(${_Ice_imported_target} PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
            IMPORTED_LOCATION "${${_Ice_component_cache}}")
        endif()
        if(EXISTS "${${_Ice_component_cache_release}}")
          set_property(TARGET ${_Ice_imported_target} APPEND PROPERTY
            IMPORTED_CONFIGURATIONS RELEASE)
          set_target_properties(${_Ice_imported_target} PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
            IMPORTED_LOCATION_RELEASE "${${_Ice_component_cache_release}}")
        endif()
        if(EXISTS "${${_Ice_component_cache_debug}}")
          set_property(TARGET ${_Ice_imported_target} APPEND PROPERTY
            IMPORTED_CONFIGURATIONS DEBUG)
          set_target_properties(${_Ice_imported_target} PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
            IMPORTED_LOCATION_DEBUG "${${_Ice_component_cache_debug}}")
        endif()
      endif()
    endif()
    unset(_Ice_component_upcase)
    unset(_Ice_component_cache)
    unset(_Ice_component_lib)
    unset(_Ice_component_found)
    unset(_Ice_imported_target)
  endforeach()
endif()

if(Ice_DEBUG)
  message(STATUS "--------FindIce.cmake results debug--------")
  message(STATUS "Ice_VERSION number: ${Ice_VERSION}")
  message(STATUS "Ice_HOME directory: ${Ice_HOME}")
  message(STATUS "Ice_INCLUDE_DIR directory: ${Ice_INCLUDE_DIR}")
  message(STATUS "Ice_SLICE_DIR directory: ${Ice_SLICE_DIR}")
  message(STATUS "Ice_LIBRARIES: ${Ice_LIBRARIES}")
  message(STATUS "Freeze_INCLUDE_DIR directory: ${Freeze_INCLUDE_DIR}")
  message(STATUS "Ice_INCLUDE_DIRS directory: ${Ice_INCLUDE_DIRS}")

  foreach(program ${_Ice_db_programs} ${_Ice_programs} ${_Ice_slice_programs})
    string(TOUPPER "${program}" program_upcase)
    message(STATUS "${program} executable: ${Ice_${program_upcase}_EXECUTABLE}")
  endforeach()

  foreach(component ${Ice_FIND_COMPONENTS})
    string(TOUPPER "${component}" component_upcase)
    set(component_lib "Ice_${component_upcase}_LIBRARIES")
    set(component_found "${component_upcase}_FOUND")
    message(STATUS "${component} library found: ${${component_found}}")
    message(STATUS "${component} library: ${${component_lib}}")
  endforeach()
  message(STATUS "----------------")
endif()

unset(_Ice_db_programs)
unset(_Ice_programs)
unset(_Ice_slice_programs)
