# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindBoost
---------

.. versionchanged:: 3.30
  This module is available only if policy :policy:`CMP0167` is not set to
  ``NEW``.  Port projects to upstream Boost's ``BoostConfig.cmake`` package
  configuration file, for which ``find_package(Boost)`` now searches.

Find Boost include dirs and libraries

Use this module by invoking :command:`find_package` with the form:

.. code-block:: cmake

  find_package(Boost
    [version] [EXACT]      # Minimum or EXACT version e.g. 1.67.0
    [REQUIRED]             # Fail with error if Boost is not found
    [COMPONENTS <libs>...] # Boost libraries by their canonical name
                           # e.g. "date_time" for "libboost_date_time"
    [OPTIONAL_COMPONENTS <libs>...]
                           # Optional Boost libraries by their canonical name)
    )                      # e.g. "date_time" for "libboost_date_time"

This module finds headers and requested component libraries OR a CMake
package configuration file provided by a "Boost CMake" build.  For the
latter case skip to the :ref:`Boost CMake` section below.

.. versionadded:: 3.7
  ``bzip2`` and ``zlib`` components (Windows only).

.. versionadded:: 3.11
  The ``OPTIONAL_COMPONENTS`` option.

.. versionadded:: 3.13
  ``stacktrace_*`` components.

.. versionadded:: 3.19
  ``bzip2`` and ``zlib`` components on all platforms.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Boost_FOUND``
  Boolean indicating whether headers and requested libraries were found.

``Boost_INCLUDE_DIRS``
  Boost include directories.

``Boost_LIBRARY_DIRS``
  Link directories for Boost libraries.

``Boost_LIBRARIES``
  Boost component libraries to be linked.

``Boost_<COMPONENT>_FOUND``
  Boolean indicating whether the component ``<COMPONENT>`` was found
  (``<COMPONENT>`` name is upper-case).

``Boost_<COMPONENT>_LIBRARY``
  Libraries to link for component ``<COMPONENT>`` (may include
  :command:`target_link_libraries` debug/optimized keywords).

``Boost_VERSION_MACRO``
  ``BOOST_VERSION`` value from ``boost/version.hpp``.

``Boost_VERSION_STRING``
  Boost version number in ``X.Y.Z`` format.

``Boost_VERSION``
  Boost version number in ``X.Y.Z`` format (same as ``Boost_VERSION_STRING``).

  .. versionchanged:: 3.15
    In previous CMake versions, this variable used the raw version string
    from the Boost header (same as ``Boost_VERSION_MACRO``).
    See policy :policy:`CMP0093`.

``Boost_LIB_VERSION``
  Version string appended to library filenames.

``Boost_VERSION_MAJOR``, ``Boost_MAJOR_VERSION``
  Boost major version number (``X`` in ``X.Y.Z``).

``Boost_VERSION_MINOR``, ``Boost_MINOR_VERSION``
  Boost minor version number (``Y`` in ``X.Y.Z``).

``Boost_VERSION_PATCH``, ``Boost_SUBMINOR_VERSION``
  Boost subminor version number (``Z`` in ``X.Y.Z``).

``Boost_VERSION_COUNT``
  Amount of version components (3).

``Boost_LIB_DIAGNOSTIC_DEFINITIONS`` (Windows-specific)
  Pass to :command:`add_definitions` to have diagnostic
  information about Boost's automatic linking
  displayed during compilation

.. versionadded:: 3.15
  The ``Boost_VERSION_<PART>`` variables.

Cache Variables
^^^^^^^^^^^^^^^

Search results are saved persistently in CMake cache entries:

``Boost_INCLUDE_DIR``
  Directory containing Boost headers.

``Boost_LIBRARY_DIR_RELEASE``
  Directory containing release Boost libraries.

``Boost_LIBRARY_DIR_DEBUG``
  Directory containing debug Boost libraries.

``Boost_<COMPONENT>_LIBRARY_DEBUG``
  Component ``<COMPONENT>`` library debug variant.

``Boost_<COMPONENT>_LIBRARY_RELEASE``
  Component ``<COMPONENT>`` library release variant.

.. versionadded:: 3.3
  Per-configuration variables ``Boost_LIBRARY_DIR_RELEASE`` and
  ``Boost_LIBRARY_DIR_DEBUG``.

Hints
^^^^^

This module reads hints about search locations from variables:

``BOOST_ROOT``, ``BOOSTROOT``
  Preferred installation prefix.

``BOOST_INCLUDEDIR``
  Preferred include directory e.g. ``<prefix>/include``.

``BOOST_LIBRARYDIR``
  Preferred library directory e.g. ``<prefix>/lib``.

``Boost_NO_SYSTEM_PATHS``
  Set to ``ON`` to disable searching in locations not
  specified by these hint variables. Default is ``OFF``.

``Boost_ADDITIONAL_VERSIONS``
  List of Boost versions not known to this module.
  (Boost install locations may contain the version).

Users may set these hints or results as ``CACHE`` entries.  Projects
should not read these entries directly but instead use the above
result variables.  Note that some hint names start in upper-case
``BOOST``.  One may specify these as environment variables if they are
not specified as CMake variables or cache entries.

This module first searches for the Boost header files using the above
hint variables (excluding ``BOOST_LIBRARYDIR``) and saves the result in
``Boost_INCLUDE_DIR``.  Then it searches for requested component libraries
using the above hints (excluding ``BOOST_INCLUDEDIR`` and
``Boost_ADDITIONAL_VERSIONS``), "lib" directories near ``Boost_INCLUDE_DIR``,
and the library name configuration settings below.  It saves the
library directories in ``Boost_LIBRARY_DIR_DEBUG`` and
``Boost_LIBRARY_DIR_RELEASE`` and individual library
locations in ``Boost_<COMPONENT>_LIBRARY_DEBUG`` and ``Boost_<COMPONENT>_LIBRARY_RELEASE``.
When one changes settings used by previous searches in the same build
tree (excluding environment variables) this module discards previous
search results affected by the changes and searches again.

Imported Targets
^^^^^^^^^^^^^^^^

.. versionadded:: 3.5

This module provides the following :ref:`Imported Targets`:

``Boost::boost``
  Target for header-only dependencies. (Boost include directory).

``Boost::headers``
  .. versionadded:: 3.15
    Alias for ``Boost::boost``.

``Boost::<component>``
  Target for specific component dependency (shared or static library);
  ``<component>`` name is lower-case.

``Boost::diagnostic_definitions``
  Interface target to enable diagnostic information about Boost's automatic
  linking during compilation (adds ``-DBOOST_LIB_DIAGNOSTIC``).

``Boost::disable_autolinking``
  Interface target to disable automatic linking with MSVC
  (adds ``-DBOOST_ALL_NO_LIB``).

``Boost::dynamic_linking``
  Interface target to enable dynamic linking with MSVC
  (adds ``-DBOOST_ALL_DYN_LINK``).

Implicit dependencies such as ``Boost::filesystem`` requiring
``Boost::system`` will be automatically detected and satisfied, even
if system is not specified when using :command:`find_package` and if
``Boost::system`` is not added to :command:`target_link_libraries`.  If using
``Boost::thread``, then ``Threads::Threads`` will also be added automatically.

It is important to note that the imported targets behave differently
than variables created by this module: multiple calls to
:command:`find_package(Boost)` in the same directory or sub-directories with
different options (e.g. static or shared) will not override the
values of the targets created by the first call.

Other Variables
^^^^^^^^^^^^^^^

Boost libraries come in many variants encoded in their file name.
Users or projects may tell this module which variant to find by
setting variables:

``Boost_USE_DEBUG_LIBS``
  .. versionadded:: 3.10

  Set to ``ON`` or ``OFF`` to specify whether to search and use the debug
  libraries.  Default is ``ON``.

``Boost_USE_RELEASE_LIBS``
  .. versionadded:: 3.10

  Set to ``ON`` or ``OFF`` to specify whether to search and use the release
  libraries.  Default is ``ON``.

``Boost_USE_MULTITHREADED``
  Set to OFF to use the non-multithreaded libraries ("mt" tag). Default is
  ``ON``.

``Boost_USE_STATIC_LIBS``
  Set to ON to force the use of the static libraries.  Default is ``OFF``.

``Boost_USE_STATIC_RUNTIME``
  Set to ``ON`` or ``OFF`` to specify whether to use libraries linked
  statically to the C++ runtime ("s" tag).  Default is platform dependent.

``Boost_USE_DEBUG_RUNTIME``
  Set to ``ON`` or ``OFF`` to specify whether to use libraries linked to the
  MS debug C++ runtime ("g" tag).  Default is ``ON``.

``Boost_USE_DEBUG_PYTHON``
  Set to ``ON`` to use libraries compiled with a debug Python build ("y"
  tag).  Default is ``OFF``.

``Boost_USE_STLPORT``
  Set to ``ON`` to use libraries compiled with STLPort ("p" tag). Default is
  ``OFF``.

``Boost_USE_STLPORT_DEPRECATED_NATIVE_IOSTREAMS``
  Set to ON to use libraries compiled with STLPort deprecated "native
  iostreams" ("n" tag).  Default is ``OFF``.

``Boost_COMPILER``
  Set to the compiler-specific library suffix (e.g. ``-gcc43``).  Default is
  auto-computed for the C++ compiler in use.

  .. versionchanged:: 3.9
    A list may be used if multiple compatible suffixes should be tested for,
    in decreasing order of preference.

``Boost_LIB_PREFIX``
  .. versionadded:: 3.18

  Set to the platform-specific library name prefix (e.g. ``lib``) used by
  Boost static libs.  This is needed only on platforms where CMake does not
  know the prefix by default.

``Boost_ARCHITECTURE``
  .. versionadded:: 3.13

  Set to the architecture-specific library suffix (e.g. ``-x64``).
  Default is auto-computed for the C++ compiler in use.

``Boost_THREADAPI``
  Suffix for ``thread`` component library name, such as ``pthread`` or
  ``win32``.  Names with and without this suffix will both be tried.

``Boost_NAMESPACE``
  Alternate namespace used to build boost with e.g. if set to ``myboost``,
  will search for ``myboost_thread`` instead of ``boost_thread``.

Other variables one may set to control this module are:

``Boost_DEBUG``
  Set to ``ON`` to enable debug output from ``FindBoost``.
  Please enable this before filing any bug report.

``Boost_REALPATH``
  Set to ``ON`` to resolve symlinks for discovered libraries to assist with
  packaging.  For example, the "system" component library may be resolved to
  ``/usr/lib/libboost_system.so.1.67.0`` instead of
  ``/usr/lib/libboost_system.so``.  This does not affect linking and should
  not be enabled unless the user needs this information.

``Boost_LIBRARY_DIR``
  Default value for ``Boost_LIBRARY_DIR_RELEASE`` and
  ``Boost_LIBRARY_DIR_DEBUG``.

``Boost_NO_WARN_NEW_VERSIONS``
  .. versionadded:: 3.20

  Set to ``ON`` to suppress the warning about unknown dependencies for new
  Boost versions.

On Visual Studio and Borland compilers Boost headers request automatic
linking to corresponding libraries.  This requires matching libraries
to be linked explicitly or available in the link library search path.
In this case setting ``Boost_USE_STATIC_LIBS`` to ``OFF`` may not achieve
dynamic linking.  Boost automatic linking typically requests static
libraries with a few exceptions (such as ``Boost.Python``).  Use:

.. code-block:: cmake

  add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})

to ask Boost to report information about automatic linking requests.

Examples
^^^^^^^^

Find Boost headers only:

.. code-block:: cmake

  find_package(Boost 1.36.0)
  if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    add_executable(foo foo.cc)
  endif()

Find Boost libraries and use imported targets:

.. code-block:: cmake

  find_package(Boost 1.56 REQUIRED COMPONENTS
               date_time filesystem iostreams)
  add_executable(foo foo.cc)
  target_link_libraries(foo Boost::date_time Boost::filesystem
                            Boost::iostreams)

Find Boost Python 3.6 libraries and use imported targets:

.. code-block:: cmake

  find_package(Boost 1.67 REQUIRED COMPONENTS
               python36 numpy36)
  add_executable(foo foo.cc)
  target_link_libraries(foo Boost::python36 Boost::numpy36)

Find Boost headers and some *static* (release only) libraries:

.. code-block:: cmake

  set(Boost_USE_STATIC_LIBS        ON)  # only find static libs
  set(Boost_USE_DEBUG_LIBS        OFF)  # ignore debug libs and
  set(Boost_USE_RELEASE_LIBS       ON)  # only find release libs
  set(Boost_USE_MULTITHREADED      ON)
  set(Boost_USE_STATIC_RUNTIME    OFF)
  find_package(Boost 1.66.0 COMPONENTS date_time filesystem system ...)
  if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    add_executable(foo foo.cc)
    target_link_libraries(foo ${Boost_LIBRARIES})
  endif()

.. _`Boost CMake`:

Boost CMake
^^^^^^^^^^^

If Boost was built using the boost-cmake project or from Boost 1.70.0 on
it provides a package configuration file for use with find_package's config mode.
This module looks for the package configuration file called
``BoostConfig.cmake`` or ``boost-config.cmake`` and stores the result in
``CACHE`` entry ``Boost_DIR``.  If found, the package configuration file is loaded
and this module returns with no further action.  See documentation of
the Boost CMake package configuration for details on what it provides.

Set ``Boost_NO_BOOST_CMAKE`` to ``ON``, to disable the search for boost-cmake.
#]=======================================================================]

cmake_policy(GET CMP0167 _FindBoost_CMP0167)
if(_FindBoost_CMP0167 STREQUAL "NEW")
  message(FATAL_ERROR "The FindBoost module has been removed by policy CMP0167.")
endif()

if(_FindBoost_testing)
  set(_FindBoost_included TRUE)
  return()
endif()

# The FPHSA helper provides standard way of reporting final search results to
# the user including the version and component checks.
include(FindPackageHandleStandardArgs)

# Save project's policies
cmake_policy(PUSH)
cmake_policy(SET CMP0102 NEW) # if mark_as_advanced(non_cache_var)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

function(_boost_get_existing_target component target_var)
  set(names "${component}")
  if(component MATCHES "^([a-z_]*)(python|numpy)([1-9])\\.?([0-9]+)?$")
    # handle pythonXY and numpyXY versioned components and also python X.Y, mpi_python etc.
    list(APPEND names
      "${CMAKE_MATCH_1}${CMAKE_MATCH_2}" # python
      "${CMAKE_MATCH_1}${CMAKE_MATCH_2}${CMAKE_MATCH_3}" # pythonX
      "${CMAKE_MATCH_1}${CMAKE_MATCH_2}${CMAKE_MATCH_3}${CMAKE_MATCH_4}" #pythonXY
    )
  endif()
  # https://github.com/boost-cmake/boost-cmake uses boost::file_system etc.
  # So handle similar constructions of target names
  string(TOLOWER "${component}" lower_component)
  list(APPEND names "${lower_component}")
  foreach(prefix Boost boost)
    foreach(name IN LISTS names)
      if(TARGET "${prefix}::${name}")
        # The target may be an INTERFACE library that wraps around a single other
        # target for compatibility.  Unwrap this layer so we can extract real info.
        if("${name}" MATCHES "^(python|numpy|mpi_python)([1-9])([0-9]+)$")
          set(name_nv "${CMAKE_MATCH_1}")
          if(TARGET "${prefix}::${name_nv}")
            get_property(type TARGET "${prefix}::${name}" PROPERTY TYPE)
            if(type STREQUAL "INTERFACE_LIBRARY")
              get_property(lib TARGET "${prefix}::${name}" PROPERTY INTERFACE_LINK_LIBRARIES)
              if("${lib}" STREQUAL "${prefix}::${name_nv}")
                set(${target_var} "${prefix}::${name_nv}" PARENT_SCOPE)
                return()
              endif()
            endif()
          endif()
        endif()
        set(${target_var} "${prefix}::${name}" PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endforeach()
  set(${target_var} "" PARENT_SCOPE)
endfunction()

function(_boost_get_canonical_target_name component target_var)
  string(TOLOWER "${component}" component)
  if(component MATCHES "^([a-z_]*)(python|numpy)([1-9])\\.?([0-9]+)?$")
    # handle pythonXY and numpyXY versioned components and also python X.Y, mpi_python etc.
    set(${target_var} "Boost::${CMAKE_MATCH_1}${CMAKE_MATCH_2}" PARENT_SCOPE)
  else()
    set(${target_var} "Boost::${component}" PARENT_SCOPE)
  endif()
endfunction()

macro(_boost_set_in_parent_scope name value)
  # Set a variable in parent scope and make it visible in current scope
  set(${name} "${value}" PARENT_SCOPE)
  set(${name} "${value}")
endmacro()

macro(_boost_set_if_unset name value)
  if(NOT ${name})
    _boost_set_in_parent_scope(${name} "${value}")
  endif()
endmacro()

macro(_boost_set_cache_if_unset name value)
  if(NOT ${name})
    set(${name} "${value}" CACHE STRING "" FORCE)
  endif()
endmacro()

macro(_boost_append_include_dir target)
  get_target_property(inc "${target}" INTERFACE_INCLUDE_DIRECTORIES)
  if(inc)
    list(APPEND include_dirs "${inc}")
  endif()
endmacro()

function(_boost_set_legacy_variables_from_config)
  # Set legacy variables for compatibility if not set
  set(include_dirs "")
  set(library_dirs "")
  set(libraries "")
  # Header targets Boost::headers or Boost::boost
  foreach(comp headers boost)
    _boost_get_existing_target(${comp} target)
    if(target)
      _boost_append_include_dir("${target}")
    endif()
  endforeach()
  # Library targets
  foreach(comp IN LISTS Boost_FIND_COMPONENTS)
    string(TOUPPER ${comp} uppercomp)
    # Overwrite if set
    _boost_set_in_parent_scope(Boost_${uppercomp}_FOUND "${Boost_${comp}_FOUND}")
    if(Boost_${comp}_FOUND)
      _boost_get_existing_target(${comp} target)
      if(NOT target)
        if(Boost_DEBUG OR Boost_VERBOSE)
          message(WARNING "Could not find imported target for required component '${comp}'. Legacy variables for this component might be missing. Refer to the documentation of your Boost installation for help on variables to use.")
        endif()
        continue()
      endif()
      _boost_append_include_dir("${target}")
      _boost_set_if_unset(Boost_${uppercomp}_LIBRARY "${target}")
      _boost_set_if_unset(Boost_${uppercomp}_LIBRARIES "${target}") # Very old legacy variable
      list(APPEND libraries "${target}")
      get_property(type TARGET "${target}" PROPERTY TYPE)
      if(NOT type STREQUAL "INTERFACE_LIBRARY")
        foreach(cfg RELEASE DEBUG)
          get_target_property(lib ${target} IMPORTED_LOCATION_${cfg})
          if(lib)
            get_filename_component(lib_dir "${lib}" DIRECTORY)
            list(APPEND library_dirs ${lib_dir})
            _boost_set_cache_if_unset(Boost_${uppercomp}_LIBRARY_${cfg} "${lib}")
          endif()
        endforeach()
      elseif(Boost_DEBUG OR Boost_VERBOSE)
        # For projects using only the Boost::* targets this warning can be safely ignored.
        message(WARNING "Imported target '${target}' for required component '${comp}' has no artifact. Legacy variables for this component might be missing. Refer to the documentation of your Boost installation for help on variables to use.")
      endif()
      _boost_get_canonical_target_name("${comp}" canonical_target)
      if(NOT TARGET "${canonical_target}")
        add_library("${canonical_target}" INTERFACE IMPORTED)
        target_link_libraries("${canonical_target}" INTERFACE "${target}")
      endif()
    endif()
  endforeach()
  list(REMOVE_DUPLICATES include_dirs)
  list(REMOVE_DUPLICATES library_dirs)
  _boost_set_if_unset(Boost_INCLUDE_DIRS "${include_dirs}")
  _boost_set_if_unset(Boost_LIBRARY_DIRS "${library_dirs}")
  _boost_set_if_unset(Boost_LIBRARIES "${libraries}")
  _boost_set_if_unset(Boost_VERSION_STRING "${Boost_VERSION_MAJOR}.${Boost_VERSION_MINOR}.${Boost_VERSION_PATCH}")
  find_path(Boost_INCLUDE_DIR
    NAMES boost/version.hpp boost/config.hpp
    HINTS ${Boost_INCLUDE_DIRS}
    NO_DEFAULT_PATH
  )
  if(NOT Boost_VERSION_MACRO OR NOT Boost_LIB_VERSION)
    set(version_file ${Boost_INCLUDE_DIR}/boost/version.hpp)
    if(EXISTS "${version_file}")
      file(STRINGS "${version_file}" contents REGEX "#define BOOST_(LIB_)?VERSION ")
      if(contents MATCHES "#define BOOST_VERSION ([0-9]+)")
        _boost_set_if_unset(Boost_VERSION_MACRO "${CMAKE_MATCH_1}")
      endif()
      if(contents MATCHES "#define BOOST_LIB_VERSION \"([0-9_]+)\"")
        _boost_set_if_unset(Boost_LIB_VERSION "${CMAKE_MATCH_1}")
      endif()
    endif()
  endif()
  _boost_set_if_unset(Boost_MAJOR_VERSION ${Boost_VERSION_MAJOR})
  _boost_set_if_unset(Boost_MINOR_VERSION ${Boost_VERSION_MINOR})
  _boost_set_if_unset(Boost_SUBMINOR_VERSION ${Boost_VERSION_PATCH})
  if(WIN32)
    _boost_set_if_unset(Boost_LIB_DIAGNOSTIC_DEFINITIONS "-DBOOST_LIB_DIAGNOSTIC")
  endif()
  if(NOT TARGET Boost::headers)
    add_library(Boost::headers INTERFACE IMPORTED)
    target_include_directories(Boost::headers INTERFACE ${Boost_INCLUDE_DIRS})
  endif()
  # Legacy targets w/o functionality as all handled by defined targets
  foreach(lib diagnostic_definitions disable_autolinking dynamic_linking)
    if(NOT TARGET Boost::${lib})
      add_library(Boost::${lib} INTERFACE IMPORTED)
    endif()
  endforeach()
  if(NOT TARGET Boost::boost)
    add_library(Boost::boost INTERFACE IMPORTED)
    target_link_libraries(Boost::boost INTERFACE Boost::headers)
  endif()
endfunction()

#-------------------------------------------------------------------------------
# Before we go searching, check whether a boost cmake package is available, unless
# the user specifically asked NOT to search for one.
#
# If Boost_DIR is set, this behaves as any find_package call would. If not,
# it looks at BOOST_ROOT and BOOSTROOT to find Boost.
#
if (NOT Boost_NO_BOOST_CMAKE)
  # If Boost_DIR is not set, look for BOOSTROOT and BOOST_ROOT as alternatives,
  # since these are more conventional for Boost.
  if ("$ENV{Boost_DIR}" STREQUAL "")
    if (NOT "$ENV{BOOST_ROOT}" STREQUAL "")
      set(ENV{Boost_DIR} $ENV{BOOST_ROOT})
    elseif (NOT "$ENV{BOOSTROOT}" STREQUAL "")
      set(ENV{Boost_DIR} $ENV{BOOSTROOT})
    endif()
  endif()

  set(_boost_FIND_PACKAGE_ARGS "")
  if(Boost_NO_SYSTEM_PATHS)
    list(APPEND _boost_FIND_PACKAGE_ARGS NO_CMAKE_SYSTEM_PATH NO_SYSTEM_ENVIRONMENT_PATH)
  endif()

  # Do the same find_package call but look specifically for the CMake version.
  # Note that args are passed in the Boost_FIND_xxxxx variables, so there is no
  # need to delegate them to this find_package call.
  if(BOOST_ROOT AND NOT Boost_ROOT)
    # Honor BOOST_ROOT by setting Boost_ROOT with CMP0074 NEW behavior.
    cmake_policy(PUSH)
    cmake_policy(SET CMP0074 NEW)
    set(Boost_ROOT "${BOOST_ROOT}")
    set(_Boost_ROOT_FOR_CONFIG 1)
  endif()
  find_package(Boost QUIET NO_MODULE ${_boost_FIND_PACKAGE_ARGS})
  if(_Boost_ROOT_FOR_CONFIG)
    unset(_Boost_ROOT_FOR_CONFIG)
    unset(Boost_ROOT)
    cmake_policy(POP)
  endif()
  if (DEFINED Boost_DIR)
    mark_as_advanced(Boost_DIR)
  endif ()

  # If we found a boost cmake package, then we're done. Print out what we found.
  # Otherwise let the rest of the module try to find it.
  if(Boost_FOUND)
    # Convert component found variables to standard variables if required
    # Necessary for legacy boost-cmake and 1.70 builtin BoostConfig
    if(Boost_FIND_COMPONENTS)
      # Ignore the meta-component "ALL", introduced by Boost 1.73
      list(REMOVE_ITEM Boost_FIND_COMPONENTS "ALL")

      foreach(_comp IN LISTS Boost_FIND_COMPONENTS)
        if(DEFINED Boost_${_comp}_FOUND)
          continue()
        endif()
        string(TOUPPER ${_comp} _uppercomp)
        if(DEFINED Boost${_comp}_FOUND) # legacy boost-cmake project
          set(Boost_${_comp}_FOUND ${Boost${_comp}_FOUND})
        elseif(DEFINED Boost_${_uppercomp}_FOUND) # Boost 1.70
          set(Boost_${_comp}_FOUND ${Boost_${_uppercomp}_FOUND})
        endif()
      endforeach()
    endif()

    find_package_handle_standard_args(Boost HANDLE_COMPONENTS CONFIG_MODE)
    _boost_set_legacy_variables_from_config()

    # Restore project's policies
    cmake_policy(POP)
    return()
  endif()
endif()


#-------------------------------------------------------------------------------
#  FindBoost functions & macros
#

#
# Print debug text if Boost_DEBUG is set.
# Call example:
# _Boost_DEBUG_PRINT("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "debug message")
#
function(_Boost_DEBUG_PRINT file line text)
  if(Boost_DEBUG)
    message(STATUS "[ ${file}:${line} ] ${text}")
  endif()
endfunction()

#
# _Boost_DEBUG_PRINT_VAR(file line variable_name [ENVIRONMENT]
#                        [SOURCE "short explanation of origin of var value"])
#
#   ENVIRONMENT - look up environment variable instead of CMake variable
#
# Print variable name and its value if Boost_DEBUG is set.
# Call example:
# _Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" BOOST_ROOT)
#
function(_Boost_DEBUG_PRINT_VAR file line name)
  if(Boost_DEBUG)
    cmake_parse_arguments(_args "ENVIRONMENT" "SOURCE" "" ${ARGN})

    unset(source)
    if(_args_SOURCE)
      set(source " (${_args_SOURCE})")
    endif()

    if(_args_ENVIRONMENT)
      if(DEFINED ENV{${name}})
        set(value "\"$ENV{${name}}\"")
      else()
        set(value "<unset>")
      endif()
      set(_name "ENV{${name}}")
    else()
      if(DEFINED "${name}")
        set(value "\"${${name}}\"")
      else()
        set(value "<unset>")
      endif()
      set(_name "${name}")
    endif()

    _Boost_DEBUG_PRINT("${file}" "${line}" "${_name} = ${value}${source}")
  endif()
endfunction()

############################################
#
# Check the existence of the libraries.
#
############################################
# This macro was taken directly from the FindQt4.cmake file that is included
# with the CMake distribution. This is NOT my work. All work was done by the
# original authors of the FindQt4.cmake file. Only minor modifications were
# made to remove references to Qt and make this file more generally applicable
# And ELSE/ENDIF pairs were removed for readability.
#########################################################################

macro(_Boost_ADJUST_LIB_VARS basename)
  if(Boost_INCLUDE_DIR )
    if(Boost_${basename}_LIBRARY_DEBUG AND Boost_${basename}_LIBRARY_RELEASE)
      # if the generator is multi-config or if CMAKE_BUILD_TYPE is set for
      # single-config generators, set optimized and debug libraries
      get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
      if(_isMultiConfig OR CMAKE_BUILD_TYPE)
        set(Boost_${basename}_LIBRARY optimized ${Boost_${basename}_LIBRARY_RELEASE} debug ${Boost_${basename}_LIBRARY_DEBUG})
      else()
        # For single-config generators where CMAKE_BUILD_TYPE has no value,
        # just use the release libraries
        set(Boost_${basename}_LIBRARY ${Boost_${basename}_LIBRARY_RELEASE} )
      endif()
      # FIXME: This probably should be set for both cases
      set(Boost_${basename}_LIBRARIES optimized ${Boost_${basename}_LIBRARY_RELEASE} debug ${Boost_${basename}_LIBRARY_DEBUG})
    endif()

    # if only the release version was found, set the debug variable also to the release version
    if(Boost_${basename}_LIBRARY_RELEASE AND NOT Boost_${basename}_LIBRARY_DEBUG)
      set(Boost_${basename}_LIBRARY_DEBUG ${Boost_${basename}_LIBRARY_RELEASE})
      set(Boost_${basename}_LIBRARY       ${Boost_${basename}_LIBRARY_RELEASE})
      set(Boost_${basename}_LIBRARIES     ${Boost_${basename}_LIBRARY_RELEASE})
    endif()

    # if only the debug version was found, set the release variable also to the debug version
    if(Boost_${basename}_LIBRARY_DEBUG AND NOT Boost_${basename}_LIBRARY_RELEASE)
      set(Boost_${basename}_LIBRARY_RELEASE ${Boost_${basename}_LIBRARY_DEBUG})
      set(Boost_${basename}_LIBRARY         ${Boost_${basename}_LIBRARY_DEBUG})
      set(Boost_${basename}_LIBRARIES       ${Boost_${basename}_LIBRARY_DEBUG})
    endif()

    # If the debug & release library ends up being the same, omit the keywords
    if("${Boost_${basename}_LIBRARY_RELEASE}" STREQUAL "${Boost_${basename}_LIBRARY_DEBUG}")
      set(Boost_${basename}_LIBRARY   ${Boost_${basename}_LIBRARY_RELEASE} )
      set(Boost_${basename}_LIBRARIES ${Boost_${basename}_LIBRARY_RELEASE} )
    endif()

    if(Boost_${basename}_LIBRARY AND Boost_${basename}_HEADER)
      set(Boost_${basename}_FOUND ON)
      if("x${basename}" STREQUAL "xTHREAD" AND NOT TARGET Threads::Threads)
        string(APPEND Boost_ERROR_REASON_THREAD " (missing dependency: Threads)")
        set(Boost_THREAD_FOUND OFF)
      endif()
    endif()

  endif()
  # Make variables changeable to the advanced user
  mark_as_advanced(
      Boost_${basename}_LIBRARY_RELEASE
      Boost_${basename}_LIBRARY_DEBUG
  )
endmacro()

# Detect changes in used variables.
# Compares the current variable value with the last one.
# In short form:
# v != v_LAST                      -> CHANGED = 1
# v is defined, v_LAST not         -> CHANGED = 1
# v is not defined, but v_LAST is  -> CHANGED = 1
# otherwise                        -> CHANGED = 0
# CHANGED is returned in variable named ${changed_var}
macro(_Boost_CHANGE_DETECT changed_var)
  set(${changed_var} 0)
  foreach(v ${ARGN})
    if(DEFINED _Boost_COMPONENTS_SEARCHED)
      if(${v})
        if(_${v}_LAST)
          string(COMPARE NOTEQUAL "${${v}}" "${_${v}_LAST}" _${v}_CHANGED)
        else()
          set(_${v}_CHANGED 1)
        endif()
      elseif(_${v}_LAST)
        set(_${v}_CHANGED 1)
      endif()
      if(_${v}_CHANGED)
        set(${changed_var} 1)
      endif()
    else()
      set(_${v}_CHANGED 0)
    endif()
  endforeach()
endmacro()

#
# Find the given library (var).
# Use 'build_type' to support different lib paths for RELEASE or DEBUG builds
#
macro(_Boost_FIND_LIBRARY var build_type)

  find_library(${var} ${ARGN})

  if(${var})
    # If this is the first library found then save Boost_LIBRARY_DIR_[RELEASE,DEBUG].
    if(NOT Boost_LIBRARY_DIR_${build_type})
      get_filename_component(_dir "${${var}}" PATH)
      set(Boost_LIBRARY_DIR_${build_type} "${_dir}" CACHE PATH "Boost library directory ${build_type}" FORCE)
    endif()
  elseif(_Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT)
    # Try component-specific hints but do not save Boost_LIBRARY_DIR_[RELEASE,DEBUG].
    find_library(${var} HINTS ${_Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT} ${ARGN})
  endif()

  # If Boost_LIBRARY_DIR_[RELEASE,DEBUG] is known then search only there.
  if(Boost_LIBRARY_DIR_${build_type})
    set(_boost_LIBRARY_SEARCH_DIRS_${build_type} ${Boost_LIBRARY_DIR_${build_type}} NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    _Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                           "Boost_LIBRARY_DIR_${build_type}")
    _Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                           "_boost_LIBRARY_SEARCH_DIRS_${build_type}")
  endif()
endmacro()

#-------------------------------------------------------------------------------

# Convert CMAKE_CXX_COMPILER_VERSION to boost compiler suffix version.
function(_Boost_COMPILER_DUMPVERSION _OUTPUT_VERSION _OUTPUT_VERSION_MAJOR _OUTPUT_VERSION_MINOR)
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+)(\\.[0-9]+)?" "\\1"
    _boost_COMPILER_VERSION_MAJOR "${CMAKE_CXX_COMPILER_VERSION}")
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+)(\\.[0-9]+)?" "\\2"
    _boost_COMPILER_VERSION_MINOR "${CMAKE_CXX_COMPILER_VERSION}")

  set(_boost_COMPILER_VERSION "${_boost_COMPILER_VERSION_MAJOR}${_boost_COMPILER_VERSION_MINOR}")

  set(${_OUTPUT_VERSION} ${_boost_COMPILER_VERSION} PARENT_SCOPE)
  set(${_OUTPUT_VERSION_MAJOR} ${_boost_COMPILER_VERSION_MAJOR} PARENT_SCOPE)
  set(${_OUTPUT_VERSION_MINOR} ${_boost_COMPILER_VERSION_MINOR} PARENT_SCOPE)
endfunction()

#
# Take a list of libraries with "thread" in it
# and prepend duplicates with "thread_${Boost_THREADAPI}"
# at the front of the list
#
function(_Boost_PREPEND_LIST_WITH_THREADAPI _output)
  set(_orig_libnames ${ARGN})
  string(REPLACE "thread" "thread_${Boost_THREADAPI}" _threadapi_libnames "${_orig_libnames}")
  set(${_output} ${_threadapi_libnames} ${_orig_libnames} PARENT_SCOPE)
endfunction()

#
# If a library is found, replace its cache entry with its REALPATH
#
function(_Boost_SWAP_WITH_REALPATH _library _docstring)
  if(${_library})
    get_filename_component(_boost_filepathreal ${${_library}} REALPATH)
    unset(${_library} CACHE)
    set(${_library} ${_boost_filepathreal} CACHE FILEPATH "${_docstring}")
  endif()
endfunction()

function(_Boost_CHECK_SPELLING _var)
  if(${_var})
    string(TOUPPER ${_var} _var_UC)
    message(FATAL_ERROR "ERROR: ${_var} is not the correct spelling.  The proper spelling is ${_var_UC}.")
  endif()
endfunction()

# Guesses Boost's compiler prefix used in built library names
# Returns the guess by setting the variable pointed to by _ret
function(_Boost_GUESS_COMPILER_PREFIX _ret)
  if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xIntel"
      OR "x${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" STREQUAL "xIntelLLVM")
    if(WIN32)
      set (_boost_COMPILER "-iw")
    else()
      set (_boost_COMPILER "-il")
    endif()
  elseif (GHSMULTI)
    set(_boost_COMPILER "-ghs")
  elseif("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC" OR "x${CMAKE_CXX_SIMULATE_ID}" STREQUAL "xMSVC")
    if(MSVC_TOOLSET_VERSION GREATER_EQUAL 150)
      # Not yet known.
      set(_boost_COMPILER "")
    elseif(MSVC_TOOLSET_VERSION GREATER_EQUAL 140)
      # MSVC toolset 14.x versions are forward compatible.
      set(_boost_COMPILER "")
      foreach(v 9 8 7 6 5 4 3 2 1 0)
        if(MSVC_TOOLSET_VERSION GREATER_EQUAL 14${v})
          list(APPEND _boost_COMPILER "-vc14${v}")
        endif()
      endforeach()
    elseif(MSVC_TOOLSET_VERSION GREATER_EQUAL 80)
      set(_boost_COMPILER "-vc${MSVC_TOOLSET_VERSION}")
    elseif(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.10)
      set(_boost_COMPILER "-vc71")
    elseif(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13) # Good luck!
      set(_boost_COMPILER "-vc7") # yes, this is correct
    else() # VS 6.0 Good luck!
      set(_boost_COMPILER "-vc6") # yes, this is correct
    endif()

    if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang")
      string(REPLACE "." ";" VERSION_LIST "${CMAKE_CXX_COMPILER_VERSION}")
      list(GET VERSION_LIST 0 CLANG_VERSION_MAJOR)
      set(_boost_COMPILER "-clangw${CLANG_VERSION_MAJOR};${_boost_COMPILER}")
    endif()
  elseif (BORLAND)
    set(_boost_COMPILER "-bcb")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "SunPro")
    set(_boost_COMPILER "-sw")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "XL")
    set(_boost_COMPILER "-xlc")
  elseif (MINGW)
    if(Boost_VERSION_STRING VERSION_LESS 1.34)
        set(_boost_COMPILER "-mgw") # no GCC version encoding prior to 1.34
    else()
      _Boost_COMPILER_DUMPVERSION(_boost_COMPILER_VERSION _boost_COMPILER_VERSION_MAJOR _boost_COMPILER_VERSION_MINOR)
      if(Boost_VERSION_STRING VERSION_GREATER_EQUAL 1.73 AND _boost_COMPILER_VERSION_MAJOR VERSION_GREATER_EQUAL 5)
        set(_boost_COMPILER "-mgw${_boost_COMPILER_VERSION_MAJOR}")
      else()
        set(_boost_COMPILER "-mgw${_boost_COMPILER_VERSION}")
      endif()
    endif()
  elseif (UNIX)
    _Boost_COMPILER_DUMPVERSION(_boost_COMPILER_VERSION _boost_COMPILER_VERSION_MAJOR _boost_COMPILER_VERSION_MINOR)
    if(NOT Boost_VERSION_STRING VERSION_LESS 1.69.0)
      # From GCC 5 and clang 4, versioning changes and minor becomes patch.
      # For those compilers, patch is exclude from compiler tag in Boost 1.69+ library naming.
      if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND _boost_COMPILER_VERSION_MAJOR VERSION_GREATER 4) OR CMAKE_CXX_COMPILER_ID STREQUAL "LCC")
        set(_boost_COMPILER_VERSION "${_boost_COMPILER_VERSION_MAJOR}")
      elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND _boost_COMPILER_VERSION_MAJOR VERSION_GREATER 3)
        set(_boost_COMPILER_VERSION "${_boost_COMPILER_VERSION_MAJOR}")
      endif()
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "LCC")
      if(Boost_VERSION_STRING VERSION_LESS 1.34)
        set(_boost_COMPILER "-gcc") # no GCC version encoding prior to 1.34
      else()
        # Determine which version of GCC we have.
        if(APPLE)
          if(Boost_VERSION_STRING VERSION_LESS 1.36.0)
            # In Boost <= 1.35.0, there is no mangled compiler name for
            # the macOS/Darwin version of GCC.
            set(_boost_COMPILER "")
          else()
            # In Boost 1.36.0 and newer, the mangled compiler name used
            # on macOS/Darwin is "xgcc".
            set(_boost_COMPILER "-xgcc${_boost_COMPILER_VERSION}")
          endif()
        else()
          set(_boost_COMPILER "-gcc${_boost_COMPILER_VERSION}")
        endif()
      endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      # TODO: Find out any Boost version constraints vs clang support.
      set(_boost_COMPILER "-clang${_boost_COMPILER_VERSION}")
    endif()
  else()
    set(_boost_COMPILER "")
  endif()
  _Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                         "_boost_COMPILER" SOURCE "guessed")
  set(${_ret} ${_boost_COMPILER} PARENT_SCOPE)
endfunction()

#
# Get component dependencies.  Requires the dependencies to have been
# defined for the Boost release version.
#
# component - the component to check
# _ret - list of library dependencies
#
function(_Boost_COMPONENT_DEPENDENCIES component _ret)
  # Note: to add a new Boost release, run
  #
  #   % cmake -DBOOST_DIR=/path/to/boost/source -P Utilities/Scripts/BoostScanDeps.cmake
  #
  # The output may be added in a new block below.  If it's the same as
  # the previous release, simply update the version range of the block
  # for the previous release.  Also check if any new components have
  # been added, and add any new components to
  # _Boost_COMPONENT_HEADERS.
  #
  # This information was originally generated by running
  # BoostScanDeps.cmake against every boost release to date supported
  # by FindBoost:
  #
  #   % for version in /path/to/boost/sources/*
  #     do
  #       cmake -DBOOST_DIR=$version -P Utilities/Scripts/BoostScanDeps.cmake
  #     done
  #
  # The output was then updated by search and replace with these regexes:
  #
  # - Strip message(STATUS) prefix dashes
  #   s;^-- ;;
  # - Indent
  #   s;^set(;    set(;;
  # - Add conditionals
  #   s;Scanning /path/to/boost/sources/boost_\(.*\)_\(.*\)_\(.*);  elseif(NOT Boost_VERSION_STRING VERSION_LESS \1\.\2\.\3 AND Boost_VERSION_STRING VERSION_LESS xxxx);
  #
  # This results in the logic seen below, but will require the xxxx
  # replacing with the following Boost release version (or the next
  # minor version to be released, e.g. 1.59 was the latest at the time
  # of writing, making 1.60 the next. Identical consecutive releases
  # were then merged together by updating the end range of the first
  # block and removing the following redundant blocks.
  #
  # Running the script against all historical releases should be
  # required only if the BoostScanDeps.cmake script logic is changed.
  # The addition of a new release should only require it to be run
  # against the new release.

  # Handle Python version suffixes
  if(component MATCHES "^(python|mpi_python|numpy)([0-9][0-9]?|[0-9]\\.[0-9]+)\$")
    set(component "${CMAKE_MATCH_1}")
    set(component_python_version "${CMAKE_MATCH_2}")
  endif()

  set(_Boost_IMPORTED_TARGETS TRUE)
  if(Boost_VERSION_STRING)
    if(Boost_VERSION_STRING VERSION_LESS 1.33.0)
      message(WARNING "Imported targets and dependency information not available for Boost version ${Boost_VERSION_STRING} (all versions older than 1.33)")
      set(_Boost_IMPORTED_TARGETS FALSE)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.35.0)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex thread)
      set(_Boost_REGEX_DEPENDENCIES thread)
      set(_Boost_WAVE_DEPENDENCIES filesystem thread)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.36.0)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_WAVE_DEPENDENCIES filesystem system thread)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.38.0)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_WAVE_DEPENDENCIES filesystem system thread)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.43.0)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_THREAD_DEPENDENCIES date_time)
      set(_Boost_WAVE_DEPENDENCIES filesystem system thread date_time)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.44.0)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l random)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_THREAD_DEPENDENCIES date_time)
      set(_Boost_WAVE_DEPENDENCIES filesystem system thread date_time)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.45.0)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l random serialization)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_THREAD_DEPENDENCIES date_time)
      set(_Boost_WAVE_DEPENDENCIES serialization filesystem system thread date_time)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.47.0)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l random)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_THREAD_DEPENDENCIES date_time)
      set(_Boost_WAVE_DEPENDENCIES filesystem system serialization thread date_time)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.48.0)
      set(_Boost_CHRONO_DEPENDENCIES system)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l random)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_THREAD_DEPENDENCIES date_time)
      set(_Boost_WAVE_DEPENDENCIES filesystem system serialization thread date_time)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.50.0)
      set(_Boost_CHRONO_DEPENDENCIES system)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l random)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_THREAD_DEPENDENCIES date_time)
      set(_Boost_TIMER_DEPENDENCIES chrono system)
      set(_Boost_WAVE_DEPENDENCIES filesystem system serialization thread date_time)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.53.0)
      set(_Boost_CHRONO_DEPENDENCIES system)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l regex random)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_THREAD_DEPENDENCIES chrono system date_time)
      set(_Boost_TIMER_DEPENDENCIES chrono system)
      set(_Boost_WAVE_DEPENDENCIES filesystem system serialization thread chrono date_time)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.54.0)
      set(_Boost_ATOMIC_DEPENDENCIES thread chrono system date_time)
      set(_Boost_CHRONO_DEPENDENCIES system)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l regex random)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_THREAD_DEPENDENCIES chrono system date_time atomic)
      set(_Boost_TIMER_DEPENDENCIES chrono system)
      set(_Boost_WAVE_DEPENDENCIES filesystem system serialization thread chrono date_time)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.55.0)
      set(_Boost_ATOMIC_DEPENDENCIES thread chrono system date_time)
      set(_Boost_CHRONO_DEPENDENCIES system)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_LOG_DEPENDENCIES log_setup date_time system filesystem thread regex chrono)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l regex random)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_THREAD_DEPENDENCIES chrono system date_time atomic)
      set(_Boost_TIMER_DEPENDENCIES chrono system)
      set(_Boost_WAVE_DEPENDENCIES filesystem system serialization thread chrono date_time atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.56.0)
      set(_Boost_CHRONO_DEPENDENCIES system)
      set(_Boost_COROUTINE_DEPENDENCIES context system)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_LOG_DEPENDENCIES log_setup date_time system filesystem thread regex chrono)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l regex random)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_THREAD_DEPENDENCIES chrono system date_time atomic)
      set(_Boost_TIMER_DEPENDENCIES chrono system)
      set(_Boost_WAVE_DEPENDENCIES filesystem system serialization thread chrono date_time atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.59.0)
      set(_Boost_CHRONO_DEPENDENCIES system)
      set(_Boost_COROUTINE_DEPENDENCIES context system)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_LOG_DEPENDENCIES log_setup date_time system filesystem thread regex chrono)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l atomic)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_RANDOM_DEPENDENCIES system)
      set(_Boost_THREAD_DEPENDENCIES chrono system date_time atomic)
      set(_Boost_TIMER_DEPENDENCIES chrono system)
      set(_Boost_WAVE_DEPENDENCIES filesystem system serialization thread chrono date_time atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.60.0)
      set(_Boost_CHRONO_DEPENDENCIES system)
      set(_Boost_COROUTINE_DEPENDENCIES context system)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_LOG_DEPENDENCIES log_setup date_time system filesystem thread regex chrono atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l atomic)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_RANDOM_DEPENDENCIES system)
      set(_Boost_THREAD_DEPENDENCIES chrono system date_time atomic)
      set(_Boost_TIMER_DEPENDENCIES chrono system)
      set(_Boost_WAVE_DEPENDENCIES filesystem system serialization thread chrono date_time atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.61.0)
      set(_Boost_CHRONO_DEPENDENCIES system)
      set(_Boost_COROUTINE_DEPENDENCIES context system)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_LOG_DEPENDENCIES date_time log_setup system filesystem thread regex chrono atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l atomic)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_RANDOM_DEPENDENCIES system)
      set(_Boost_THREAD_DEPENDENCIES chrono system date_time atomic)
      set(_Boost_TIMER_DEPENDENCIES chrono system)
      set(_Boost_WAVE_DEPENDENCIES filesystem system serialization thread chrono date_time atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.62.0)
      set(_Boost_CHRONO_DEPENDENCIES system)
      set(_Boost_CONTEXT_DEPENDENCIES thread chrono system date_time)
      set(_Boost_COROUTINE_DEPENDENCIES context system)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_LOG_DEPENDENCIES date_time log_setup system filesystem thread regex chrono atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l atomic)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_RANDOM_DEPENDENCIES system)
      set(_Boost_THREAD_DEPENDENCIES chrono system date_time atomic)
      set(_Boost_WAVE_DEPENDENCIES filesystem system serialization thread chrono date_time atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.63.0)
      set(_Boost_CHRONO_DEPENDENCIES system)
      set(_Boost_CONTEXT_DEPENDENCIES thread chrono system date_time)
      set(_Boost_COROUTINE_DEPENDENCIES context system)
      set(_Boost_FIBER_DEPENDENCIES context thread chrono system date_time)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_LOG_DEPENDENCIES date_time log_setup system filesystem thread regex chrono atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l atomic)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_RANDOM_DEPENDENCIES system)
      set(_Boost_THREAD_DEPENDENCIES chrono system date_time atomic)
      set(_Boost_WAVE_DEPENDENCIES filesystem system serialization thread chrono date_time atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.65.0)
      set(_Boost_CHRONO_DEPENDENCIES system)
      set(_Boost_CONTEXT_DEPENDENCIES thread chrono system date_time)
      set(_Boost_COROUTINE_DEPENDENCIES context system)
      set(_Boost_COROUTINE2_DEPENDENCIES context fiber thread chrono system date_time)
      set(_Boost_FIBER_DEPENDENCIES context thread chrono system date_time)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_LOG_DEPENDENCIES date_time log_setup system filesystem thread regex chrono atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l atomic)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_RANDOM_DEPENDENCIES system)
      set(_Boost_THREAD_DEPENDENCIES chrono system date_time atomic)
      set(_Boost_WAVE_DEPENDENCIES filesystem system serialization thread chrono date_time atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.67.0)
      set(_Boost_CHRONO_DEPENDENCIES system)
      set(_Boost_CONTEXT_DEPENDENCIES thread chrono system date_time)
      set(_Boost_COROUTINE_DEPENDENCIES context system)
      set(_Boost_FIBER_DEPENDENCIES context thread chrono system date_time)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_LOG_DEPENDENCIES date_time log_setup system filesystem thread regex chrono atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l atomic)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_NUMPY_DEPENDENCIES python${component_python_version})
      set(_Boost_RANDOM_DEPENDENCIES system)
      set(_Boost_THREAD_DEPENDENCIES chrono system date_time atomic)
      set(_Boost_TIMER_DEPENDENCIES chrono system)
      set(_Boost_WAVE_DEPENDENCIES filesystem system serialization thread chrono date_time atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.68.0)
      set(_Boost_CHRONO_DEPENDENCIES system)
      set(_Boost_CONTEXT_DEPENDENCIES thread chrono system date_time)
      set(_Boost_COROUTINE_DEPENDENCIES context system)
      set(_Boost_FIBER_DEPENDENCIES context thread chrono system date_time)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_LOG_DEPENDENCIES date_time log_setup system filesystem thread regex chrono atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l atomic)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_NUMPY_DEPENDENCIES python${component_python_version})
      set(_Boost_RANDOM_DEPENDENCIES system)
      set(_Boost_THREAD_DEPENDENCIES chrono system date_time atomic)
      set(_Boost_TIMER_DEPENDENCIES chrono system)
      set(_Boost_WAVE_DEPENDENCIES filesystem system serialization thread chrono date_time atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.69.0)
      set(_Boost_CHRONO_DEPENDENCIES system)
      set(_Boost_CONTEXT_DEPENDENCIES thread chrono system date_time)
      set(_Boost_CONTRACT_DEPENDENCIES thread chrono system date_time)
      set(_Boost_COROUTINE_DEPENDENCIES context system)
      set(_Boost_FIBER_DEPENDENCIES context thread chrono system date_time)
      set(_Boost_FILESYSTEM_DEPENDENCIES system)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_LOG_DEPENDENCIES date_time log_setup system filesystem thread regex chrono atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l atomic)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_NUMPY_DEPENDENCIES python${component_python_version})
      set(_Boost_RANDOM_DEPENDENCIES system)
      set(_Boost_THREAD_DEPENDENCIES chrono system date_time atomic)
      set(_Boost_TIMER_DEPENDENCIES chrono system)
      set(_Boost_WAVE_DEPENDENCIES filesystem system serialization thread chrono date_time atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.70.0)
      set(_Boost_CONTRACT_DEPENDENCIES thread chrono date_time)
      set(_Boost_COROUTINE_DEPENDENCIES context)
      set(_Boost_FIBER_DEPENDENCIES context)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_LOG_DEPENDENCIES date_time log_setup filesystem thread regex chrono atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l atomic)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_NUMPY_DEPENDENCIES python${component_python_version})
      set(_Boost_THREAD_DEPENDENCIES chrono date_time atomic)
      set(_Boost_TIMER_DEPENDENCIES chrono system)
      set(_Boost_WAVE_DEPENDENCIES filesystem serialization thread chrono date_time atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.72.0)
      set(_Boost_CONTRACT_DEPENDENCIES thread chrono date_time)
      set(_Boost_COROUTINE_DEPENDENCIES context)
      set(_Boost_FIBER_DEPENDENCIES context)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_LOG_DEPENDENCIES date_time log_setup filesystem thread regex chrono atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l atomic)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_NUMPY_DEPENDENCIES python${component_python_version})
      set(_Boost_THREAD_DEPENDENCIES chrono date_time atomic)
      set(_Boost_TIMER_DEPENDENCIES chrono)
      set(_Boost_WAVE_DEPENDENCIES filesystem serialization thread chrono date_time atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.73.0)
      set(_Boost_CONTRACT_DEPENDENCIES thread chrono date_time)
      set(_Boost_COROUTINE_DEPENDENCIES context)
      set(_Boost_FIBER_DEPENDENCIES context)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_LOG_DEPENDENCIES date_time log_setup filesystem thread regex chrono atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l chrono atomic)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_NUMPY_DEPENDENCIES python${component_python_version})
      set(_Boost_THREAD_DEPENDENCIES chrono date_time atomic)
      set(_Boost_TIMER_DEPENDENCIES chrono)
      set(_Boost_WAVE_DEPENDENCIES filesystem serialization thread chrono date_time atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.75.0)
      set(_Boost_CONTRACT_DEPENDENCIES thread chrono date_time)
      set(_Boost_COROUTINE_DEPENDENCIES context)
      set(_Boost_FIBER_DEPENDENCIES context)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_LOG_DEPENDENCIES date_time log_setup filesystem thread regex chrono atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l atomic)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_NUMPY_DEPENDENCIES python${component_python_version})
      set(_Boost_THREAD_DEPENDENCIES chrono date_time atomic)
      set(_Boost_TIMER_DEPENDENCIES chrono)
      set(_Boost_WAVE_DEPENDENCIES filesystem serialization thread chrono date_time atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.77.0)
      set(_Boost_CONTRACT_DEPENDENCIES thread chrono date_time)
      set(_Boost_COROUTINE_DEPENDENCIES context)
      set(_Boost_FIBER_DEPENDENCIES context)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_JSON_DEPENDENCIES container)
      set(_Boost_LOG_DEPENDENCIES date_time log_setup filesystem thread regex chrono atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l atomic)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_NUMPY_DEPENDENCIES python${component_python_version})
      set(_Boost_THREAD_DEPENDENCIES chrono date_time atomic)
      set(_Boost_TIMER_DEPENDENCIES chrono)
      set(_Boost_WAVE_DEPENDENCIES filesystem serialization thread chrono date_time atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.78.0)
      set(_Boost_CONTRACT_DEPENDENCIES thread chrono)
      set(_Boost_COROUTINE_DEPENDENCIES context)
      set(_Boost_FIBER_DEPENDENCIES context)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_JSON_DEPENDENCIES container)
      set(_Boost_LOG_DEPENDENCIES date_time log_setup filesystem thread regex chrono atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_NUMPY_DEPENDENCIES python${component_python_version})
      set(_Boost_THREAD_DEPENDENCIES chrono atomic)
      set(_Boost_TIMER_DEPENDENCIES chrono)
      set(_Boost_WAVE_DEPENDENCIES filesystem serialization thread chrono atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.83.0)
      set(_Boost_CONTRACT_DEPENDENCIES thread chrono)
      set(_Boost_COROUTINE_DEPENDENCIES context)
      set(_Boost_FIBER_DEPENDENCIES context)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_JSON_DEPENDENCIES container)
      set(_Boost_LOG_DEPENDENCIES log_setup filesystem thread regex chrono atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_NUMPY_DEPENDENCIES python${component_python_version})
      set(_Boost_THREAD_DEPENDENCIES chrono atomic)
      set(_Boost_TIMER_DEPENDENCIES chrono)
      set(_Boost_WAVE_DEPENDENCIES filesystem serialization thread chrono atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.87.0)
      set(_Boost_CONTRACT_DEPENDENCIES thread chrono)
      set(_Boost_COROUTINE_DEPENDENCIES context)
      set(_Boost_FIBER_DEPENDENCIES context)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_JSON_DEPENDENCIES container)
      set(_Boost_LOG_DEPENDENCIES log_setup filesystem thread regex chrono atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_NUMPY_DEPENDENCIES python${component_python_version})
      set(_Boost_THREAD_DEPENDENCIES chrono atomic)
      set(_Boost_WAVE_DEPENDENCIES filesystem serialization thread chrono atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    elseif(Boost_VERSION_STRING VERSION_LESS 1.88.0)
      set(_Boost_CONTRACT_DEPENDENCIES thread chrono)
      set(_Boost_COROUTINE_DEPENDENCIES context)
      set(_Boost_FIBER_DEPENDENCIES context)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_JSON_DEPENDENCIES container)
      set(_Boost_LOG_DEPENDENCIES log_setup filesystem thread regex atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_NUMPY_DEPENDENCIES python${component_python_version})
      set(_Boost_THREAD_DEPENDENCIES chrono atomic)
      set(_Boost_WAVE_DEPENDENCIES filesystem serialization thread chrono atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
    else()
      set(_Boost_CONTRACT_DEPENDENCIES thread chrono)
      set(_Boost_COROUTINE_DEPENDENCIES context)
      set(_Boost_FIBER_DEPENDENCIES context)
      set(_Boost_IOSTREAMS_DEPENDENCIES regex)
      set(_Boost_JSON_DEPENDENCIES container)
      set(_Boost_LOG_DEPENDENCIES log_setup filesystem thread regex atomic)
      set(_Boost_MATH_DEPENDENCIES math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l)
      set(_Boost_MPI_DEPENDENCIES serialization)
      set(_Boost_MPI_PYTHON_DEPENDENCIES python${component_python_version} mpi serialization)
      set(_Boost_NUMPY_DEPENDENCIES python${component_python_version})
      set(_Boost_PROCESS_DEPENDENCIES filesystem)
      set(_Boost_THREAD_DEPENDENCIES chrono atomic)
      set(_Boost_WAVE_DEPENDENCIES filesystem serialization thread chrono atomic)
      set(_Boost_WSERIALIZATION_DEPENDENCIES serialization)
      if(Boost_VERSION_STRING VERSION_GREATER_EQUAL 1.89.0 AND NOT Boost_NO_WARN_NEW_VERSIONS)
        message(WARNING "New Boost version may have incorrect or missing dependencies and imported targets")
      endif()
    endif()
  endif()

  string(TOUPPER ${component} uppercomponent)
  set(${_ret} ${_Boost_${uppercomponent}_DEPENDENCIES} PARENT_SCOPE)
  set(_Boost_IMPORTED_TARGETS ${_Boost_IMPORTED_TARGETS} PARENT_SCOPE)

  string(REGEX REPLACE ";" " " _boost_DEPS_STRING "${_Boost_${uppercomponent}_DEPENDENCIES}")
  if (NOT _boost_DEPS_STRING)
    set(_boost_DEPS_STRING "(none)")
  endif()
  # message(STATUS "Dependencies for Boost::${component}: ${_boost_DEPS_STRING}")
endfunction()

#
# Get component headers.  This is the primary header (or headers) for
# a given component, and is used to check that the headers are present
# as well as the library itself as an extra sanity check of the build
# environment.
#
# component - the component to check
# _hdrs
#
function(_Boost_COMPONENT_HEADERS component _hdrs)
  # Handle Python version suffixes
  if(component MATCHES "^(python|mpi_python|numpy)([0-9]+|[0-9]\\.[0-9]+)\$")
    set(component "${CMAKE_MATCH_1}")
    set(component_python_version "${CMAKE_MATCH_2}")
  endif()

  # Note: new boost components will require adding here.  The header
  # must be present in all versions of Boost providing a library.
  set(_Boost_ATOMIC_HEADERS              "boost/atomic.hpp")
  set(_Boost_CHRONO_HEADERS              "boost/chrono.hpp")
  set(_Boost_CONTAINER_HEADERS           "boost/container/container_fwd.hpp")
  set(_Boost_CONTRACT_HEADERS            "boost/contract.hpp")
  if(Boost_VERSION_STRING VERSION_LESS 1.61.0)
    set(_Boost_CONTEXT_HEADERS           "boost/context/all.hpp")
  else()
    set(_Boost_CONTEXT_HEADERS           "boost/context/detail/fcontext.hpp")
  endif()
  set(_Boost_COROUTINE_HEADERS           "boost/coroutine/all.hpp")
  set(_Boost_DATE_TIME_HEADERS           "boost/date_time/date.hpp")
  set(_Boost_EXCEPTION_HEADERS           "boost/exception/exception.hpp")
  set(_Boost_FIBER_HEADERS               "boost/fiber/all.hpp")
  set(_Boost_FILESYSTEM_HEADERS          "boost/filesystem/path.hpp")
  set(_Boost_GRAPH_HEADERS               "boost/graph/adjacency_list.hpp")
  set(_Boost_GRAPH_PARALLEL_HEADERS      "boost/graph/adjacency_list.hpp")
  set(_Boost_IOSTREAMS_HEADERS           "boost/iostreams/stream.hpp")
  set(_Boost_LOCALE_HEADERS              "boost/locale.hpp")
  set(_Boost_LOG_HEADERS                 "boost/log/core.hpp")
  set(_Boost_LOG_SETUP_HEADERS           "boost/log/detail/setup_config.hpp")
  set(_Boost_JSON_HEADERS                "boost/json.hpp")
  set(_Boost_MATH_HEADERS                "boost/math_fwd.hpp")
  set(_Boost_MATH_C99_HEADERS            "boost/math/tr1.hpp")
  set(_Boost_MATH_C99F_HEADERS           "boost/math/tr1.hpp")
  set(_Boost_MATH_C99L_HEADERS           "boost/math/tr1.hpp")
  set(_Boost_MATH_TR1_HEADERS            "boost/math/tr1.hpp")
  set(_Boost_MATH_TR1F_HEADERS           "boost/math/tr1.hpp")
  set(_Boost_MATH_TR1L_HEADERS           "boost/math/tr1.hpp")
  set(_Boost_MPI_HEADERS                 "boost/mpi.hpp")
  set(_Boost_MPI_PYTHON_HEADERS          "boost/mpi/python/config.hpp")
  set(_Boost_MYSQL_HEADERS               "boost/mysql.hpp")
  set(_Boost_NUMPY_HEADERS               "boost/python/numpy.hpp")
  set(_Boost_NOWIDE_HEADERS              "boost/nowide/cstdlib.hpp")
  set(_Boost_PRG_EXEC_MONITOR_HEADERS    "boost/test/prg_exec_monitor.hpp")
  set(_Boost_PROGRAM_OPTIONS_HEADERS     "boost/program_options.hpp")
  set(_Boost_PYTHON_HEADERS              "boost/python.hpp")
  set(_Boost_RANDOM_HEADERS              "boost/random.hpp")
  set(_Boost_REGEX_HEADERS               "boost/regex.hpp")
  set(_Boost_SERIALIZATION_HEADERS       "boost/serialization/serialization.hpp")
  set(_Boost_SIGNALS_HEADERS             "boost/signals.hpp")
  set(_Boost_STACKTRACE_ADDR2LINE_HEADERS "boost/stacktrace.hpp")
  set(_Boost_STACKTRACE_BACKTRACE_HEADERS "boost/stacktrace.hpp")
  set(_Boost_STACKTRACE_BASIC_HEADERS    "boost/stacktrace.hpp")
  set(_Boost_STACKTRACE_NOOP_HEADERS     "boost/stacktrace.hpp")
  set(_Boost_STACKTRACE_WINDBG_CACHED_HEADERS "boost/stacktrace.hpp")
  set(_Boost_STACKTRACE_WINDBG_HEADERS   "boost/stacktrace.hpp")
  set(_Boost_SYSTEM_HEADERS              "boost/system/config.hpp")
  set(_Boost_TEST_EXEC_MONITOR_HEADERS   "boost/test/test_exec_monitor.hpp")
  set(_Boost_THREAD_HEADERS              "boost/thread.hpp")
  set(_Boost_TIMER_HEADERS               "boost/timer.hpp")
  set(_Boost_TYPE_ERASURE_HEADERS        "boost/type_erasure/config.hpp")
  set(_Boost_UNIT_TEST_FRAMEWORK_HEADERS "boost/test/framework.hpp")
  set(_Boost_URL_HEADERS                 "boost/url.hpp")
  set(_Boost_WAVE_HEADERS                "boost/wave.hpp")
  set(_Boost_WSERIALIZATION_HEADERS      "boost/archive/text_wiarchive.hpp")
  set(_Boost_BZIP2_HEADERS               "boost/iostreams/filter/bzip2.hpp")
  set(_Boost_ZLIB_HEADERS                "boost/iostreams/filter/zlib.hpp")

  string(TOUPPER ${component} uppercomponent)
  set(${_hdrs} ${_Boost_${uppercomponent}_HEADERS} PARENT_SCOPE)

  string(REGEX REPLACE ";" " " _boost_HDRS_STRING "${_Boost_${uppercomponent}_HEADERS}")
  if (NOT _boost_HDRS_STRING)
    set(_boost_HDRS_STRING "(none)")
  endif()
  # message(STATUS "Headers for Boost::${component}: ${_boost_HDRS_STRING}")
endfunction()

#
# Determine if any missing dependencies require adding to the component list.
#
# Sets _Boost_${COMPONENT}_DEPENDENCIES for each required component,
# plus _Boost_IMPORTED_TARGETS (TRUE if imported targets should be
# defined; FALSE if dependency information is unavailable).
#
# componentvar - the component list variable name
# extravar - the indirect dependency list variable name
#
#
function(_Boost_MISSING_DEPENDENCIES componentvar extravar)
  # _boost_unprocessed_components - list of components requiring processing
  # _boost_processed_components - components already processed (or currently being processed)
  # _boost_new_components - new components discovered for future processing
  #
  list(APPEND _boost_unprocessed_components ${${componentvar}})

  while(_boost_unprocessed_components)
    list(APPEND _boost_processed_components ${_boost_unprocessed_components})
    foreach(component ${_boost_unprocessed_components})
      string(TOUPPER ${component} uppercomponent)
      set(${_ret} ${_Boost_${uppercomponent}_DEPENDENCIES} PARENT_SCOPE)
      _Boost_COMPONENT_DEPENDENCIES("${component}" _Boost_${uppercomponent}_DEPENDENCIES)
      set(_Boost_${uppercomponent}_DEPENDENCIES ${_Boost_${uppercomponent}_DEPENDENCIES} PARENT_SCOPE)
      set(_Boost_IMPORTED_TARGETS ${_Boost_IMPORTED_TARGETS} PARENT_SCOPE)
      foreach(componentdep ${_Boost_${uppercomponent}_DEPENDENCIES})
        if (NOT ("${componentdep}" IN_LIST _boost_processed_components OR "${componentdep}" IN_LIST _boost_new_components))
          list(APPEND _boost_new_components ${componentdep})
        endif()
      endforeach()
    endforeach()
    set(_boost_unprocessed_components ${_boost_new_components})
    unset(_boost_new_components)
  endwhile()
  set(_boost_extra_components ${_boost_processed_components})
  if(_boost_extra_components AND ${componentvar})
    list(REMOVE_ITEM _boost_extra_components ${${componentvar}})
  endif()
  set(${componentvar} ${_boost_processed_components} PARENT_SCOPE)
  set(${extravar} ${_boost_extra_components} PARENT_SCOPE)
endfunction()

#
# Some boost libraries may require particular set of compler features.
# The very first one was `boost::fiber` introduced in Boost 1.62.
# One can check required compiler features of it in
# - `${Boost_ROOT}/libs/fiber/build/Jamfile.v2`;
# - `${Boost_ROOT}/libs/context/build/Jamfile.v2`.
#
# TODO (Re)Check compiler features on (every?) release ???
# One may use the following command to get the files to check:
#
#   $ find . -name Jamfile.v2 | grep build | xargs grep -l cxx1
#
function(_Boost_COMPILER_FEATURES component _ret)
  # Boost >= 1.62
  if(NOT Boost_VERSION_STRING VERSION_LESS 1.62.0)
    set(_Boost_FIBER_COMPILER_FEATURES
        cxx_alias_templates
        cxx_auto_type
        cxx_constexpr
        cxx_defaulted_functions
        cxx_final
        cxx_lambdas
        cxx_noexcept
        cxx_nullptr
        cxx_rvalue_references
        cxx_thread_local
        cxx_variadic_templates
    )
    # Compiler feature for `context` same as for `fiber`.
    set(_Boost_CONTEXT_COMPILER_FEATURES ${_Boost_FIBER_COMPILER_FEATURES})
  endif()

  # Boost Contract library available in >= 1.67
  if(NOT Boost_VERSION_STRING VERSION_LESS 1.67.0)
    # From `libs/contract/build/boost_contract_build.jam`
    set(_Boost_CONTRACT_COMPILER_FEATURES
        cxx_lambdas
        cxx_variadic_templates
    )
  endif()

  string(TOUPPER ${component} uppercomponent)
  set(${_ret} ${_Boost_${uppercomponent}_COMPILER_FEATURES} PARENT_SCOPE)
endfunction()

#
# Update library search directory hint variable with paths used by prebuilt boost binaries.
#
# Prebuilt windows binaries (https://sourceforge.net/projects/boost/files/boost-binaries/)
# have library directories named using MSVC compiler version and architecture.
# This function would append corresponding directories if MSVC is a current compiler,
# so having `BOOST_ROOT` would be enough to specify to find everything.
#
function(_Boost_UPDATE_WINDOWS_LIBRARY_SEARCH_DIRS_WITH_PREBUILT_PATHS componentlibvar basedir)
  if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(_arch_suffix 64)
    else()
      set(_arch_suffix 32)
    endif()
    if(MSVC_TOOLSET_VERSION GREATER_EQUAL 150)
      # Not yet known.
    elseif(MSVC_TOOLSET_VERSION GREATER_EQUAL 140)
      # MSVC toolset 14.x versions are forward compatible.
      foreach(v 9 8 7 6 5 4 3 2 1 0)
        if(MSVC_TOOLSET_VERSION GREATER_EQUAL 14${v})
          list(APPEND ${componentlibvar} ${basedir}/lib${_arch_suffix}-msvc-14.${v})
        endif()
      endforeach()
    elseif(MSVC_TOOLSET_VERSION GREATER_EQUAL 80)
      math(EXPR _toolset_major_version "${MSVC_TOOLSET_VERSION} / 10")
      list(APPEND ${componentlibvar} ${basedir}/lib${_arch_suffix}-msvc-${_toolset_major_version}.0)
    endif()
    set(${componentlibvar} ${${componentlibvar}} PARENT_SCOPE)
  endif()
endfunction()

#
# End functions/macros
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# main.
#-------------------------------------------------------------------------------


# If the user sets Boost_LIBRARY_DIR, use it as the default for both
# configurations.
if(NOT Boost_LIBRARY_DIR_RELEASE AND Boost_LIBRARY_DIR)
  set(Boost_LIBRARY_DIR_RELEASE "${Boost_LIBRARY_DIR}")
endif()
if(NOT Boost_LIBRARY_DIR_DEBUG AND Boost_LIBRARY_DIR)
  set(Boost_LIBRARY_DIR_DEBUG   "${Boost_LIBRARY_DIR}")
endif()

if(NOT DEFINED Boost_USE_DEBUG_LIBS)
  set(Boost_USE_DEBUG_LIBS TRUE)
endif()
if(NOT DEFINED Boost_USE_RELEASE_LIBS)
  set(Boost_USE_RELEASE_LIBS TRUE)
endif()
if(NOT DEFINED Boost_USE_MULTITHREADED)
  set(Boost_USE_MULTITHREADED TRUE)
endif()
if(NOT DEFINED Boost_USE_DEBUG_RUNTIME)
  set(Boost_USE_DEBUG_RUNTIME TRUE)
endif()

# Check the version of Boost against the requested version.
if(Boost_FIND_VERSION AND NOT Boost_FIND_VERSION_MINOR)
  message(SEND_ERROR "When requesting a specific version of Boost, you must provide at least the major and minor version numbers, e.g., 1.34")
endif()

if(Boost_FIND_VERSION_EXACT)
  # The version may appear in a directory with or without the patch
  # level, even when the patch level is non-zero.
  set(_boost_TEST_VERSIONS
    "${Boost_FIND_VERSION_MAJOR}.${Boost_FIND_VERSION_MINOR}.${Boost_FIND_VERSION_PATCH}"
    "${Boost_FIND_VERSION_MAJOR}.${Boost_FIND_VERSION_MINOR}")
else()
  # The user has not requested an exact version.  Among known
  # versions, find those that are acceptable to the user request.
  #
  # Note: When adding a new Boost release, also update the dependency
  # information in _Boost_COMPONENT_DEPENDENCIES and
  # _Boost_COMPONENT_HEADERS.  See the instructions at the top of
  # _Boost_COMPONENT_DEPENDENCIES.
  set(_Boost_KNOWN_VERSIONS ${Boost_ADDITIONAL_VERSIONS}
    "1.88.0" "1.88" "1.87.0" "1.87" "1.86.0" "1.86" "1.85.0" "1.85" "1.84.0" "1.84"
    "1.83.0" "1.83" "1.82.0" "1.82" "1.81.0" "1.81" "1.80.0" "1.80" "1.79.0" "1.79"
    "1.78.0" "1.78" "1.77.0" "1.77" "1.76.0" "1.76" "1.75.0" "1.75" "1.74.0" "1.74"
    "1.73.0" "1.73" "1.72.0" "1.72" "1.71.0" "1.71" "1.70.0" "1.70" "1.69.0" "1.69"
    "1.68.0" "1.68" "1.67.0" "1.67" "1.66.0" "1.66" "1.65.1" "1.65.0" "1.65"
    "1.64.0" "1.64" "1.63.0" "1.63" "1.62.0" "1.62" "1.61.0" "1.61" "1.60.0" "1.60"
    "1.59.0" "1.59" "1.58.0" "1.58" "1.57.0" "1.57" "1.56.0" "1.56" "1.55.0" "1.55"
    "1.54.0" "1.54" "1.53.0" "1.53" "1.52.0" "1.52" "1.51.0" "1.51"
    "1.50.0" "1.50" "1.49.0" "1.49" "1.48.0" "1.48" "1.47.0" "1.47" "1.46.1"
    "1.46.0" "1.46" "1.45.0" "1.45" "1.44.0" "1.44" "1.43.0" "1.43" "1.42.0" "1.42"
    "1.41.0" "1.41" "1.40.0" "1.40" "1.39.0" "1.39" "1.38.0" "1.38" "1.37.0" "1.37"
    "1.36.1" "1.36.0" "1.36" "1.35.1" "1.35.0" "1.35" "1.34.1" "1.34.0"
    "1.34" "1.33.1" "1.33.0" "1.33")

  set(_boost_TEST_VERSIONS)
  if(Boost_FIND_VERSION)
    set(_Boost_FIND_VERSION_SHORT "${Boost_FIND_VERSION_MAJOR}.${Boost_FIND_VERSION_MINOR}")
    # Select acceptable versions.
    foreach(version ${_Boost_KNOWN_VERSIONS})
      if(NOT "${version}" VERSION_LESS "${Boost_FIND_VERSION}")
        # This version is high enough.
        list(APPEND _boost_TEST_VERSIONS "${version}")
      elseif("${version}.99" VERSION_EQUAL "${_Boost_FIND_VERSION_SHORT}.99")
        # This version is a short-form for the requested version with
        # the patch level dropped.
        list(APPEND _boost_TEST_VERSIONS "${version}")
      endif()
    endforeach()
  else()
    # Any version is acceptable.
    set(_boost_TEST_VERSIONS "${_Boost_KNOWN_VERSIONS}")
  endif()
endif()

_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_boost_TEST_VERSIONS")
_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "Boost_USE_MULTITHREADED")
_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "Boost_USE_STATIC_LIBS")
_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "Boost_USE_STATIC_RUNTIME")
_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "Boost_ADDITIONAL_VERSIONS")
_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "Boost_NO_SYSTEM_PATHS")

cmake_policy(GET CMP0074 _Boost_CMP0074)
if(NOT "x${_Boost_CMP0074}x" STREQUAL "xNEWx")
  _Boost_CHECK_SPELLING(Boost_ROOT)
endif()
unset(_Boost_CMP0074)
_Boost_CHECK_SPELLING(Boost_LIBRARYDIR)
_Boost_CHECK_SPELLING(Boost_INCLUDEDIR)

# Collect environment variable inputs as hints.  Do not consider changes.
foreach(v BOOSTROOT BOOST_ROOT BOOST_INCLUDEDIR BOOST_LIBRARYDIR)
  set(_env $ENV{${v}})
  if(_env)
    file(TO_CMAKE_PATH "${_env}" _ENV_${v})
  else()
    set(_ENV_${v} "")
  endif()
endforeach()
if(NOT _ENV_BOOST_ROOT AND _ENV_BOOSTROOT)
  set(_ENV_BOOST_ROOT "${_ENV_BOOSTROOT}")
endif()

# Collect inputs and cached results.  Detect changes since the last run.
if(NOT BOOST_ROOT AND BOOSTROOT)
  set(BOOST_ROOT "${BOOSTROOT}")
endif()
set(_Boost_VARS_DIR
  BOOST_ROOT
  Boost_NO_SYSTEM_PATHS
  )

_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "BOOST_ROOT")
_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "BOOST_ROOT" ENVIRONMENT)
_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "BOOST_INCLUDEDIR")
_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "BOOST_INCLUDEDIR" ENVIRONMENT)
_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "BOOST_LIBRARYDIR")
_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "BOOST_LIBRARYDIR" ENVIRONMENT)

# ------------------------------------------------------------------------
#  Search for Boost include DIR
# ------------------------------------------------------------------------

set(_Boost_VARS_INC BOOST_INCLUDEDIR Boost_INCLUDE_DIR Boost_ADDITIONAL_VERSIONS)
_Boost_CHANGE_DETECT(_Boost_CHANGE_INCDIR ${_Boost_VARS_DIR} ${_Boost_VARS_INC})
# Clear Boost_INCLUDE_DIR if it did not change but other input affecting the
# location did.  We will find a new one based on the new inputs.
if(_Boost_CHANGE_INCDIR AND NOT _Boost_INCLUDE_DIR_CHANGED)
  unset(Boost_INCLUDE_DIR CACHE)
endif()

if(NOT Boost_INCLUDE_DIR)
  set(_boost_INCLUDE_SEARCH_DIRS "")
  if(BOOST_INCLUDEDIR)
    list(APPEND _boost_INCLUDE_SEARCH_DIRS ${BOOST_INCLUDEDIR})
  elseif(_ENV_BOOST_INCLUDEDIR)
    list(APPEND _boost_INCLUDE_SEARCH_DIRS ${_ENV_BOOST_INCLUDEDIR})
  endif()

  if( BOOST_ROOT )
    list(APPEND _boost_INCLUDE_SEARCH_DIRS ${BOOST_ROOT}/include ${BOOST_ROOT})
  elseif( _ENV_BOOST_ROOT )
    list(APPEND _boost_INCLUDE_SEARCH_DIRS ${_ENV_BOOST_ROOT}/include ${_ENV_BOOST_ROOT})
  endif()

  if( Boost_NO_SYSTEM_PATHS)
    list(APPEND _boost_INCLUDE_SEARCH_DIRS NO_CMAKE_SYSTEM_PATH NO_SYSTEM_ENVIRONMENT_PATH)
  else()
    if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
      foreach(ver ${_boost_TEST_VERSIONS})
        string(REPLACE "." "_" ver "${ver}")
        list(APPEND _boost_INCLUDE_SEARCH_DIRS PATHS "C:/local/boost_${ver}")
      endforeach()
    endif()
    list(APPEND _boost_INCLUDE_SEARCH_DIRS PATHS
      C:/boost/include
      C:/boost
      /sw/local/include
      )
  endif()

  # Try to find Boost by stepping backwards through the Boost versions
  # we know about.
  # Build a list of path suffixes for each version.
  set(_boost_PATH_SUFFIXES)
  foreach(_boost_VER ${_boost_TEST_VERSIONS})
    # Add in a path suffix, based on the required version, ideally
    # we could read this from version.hpp, but for that to work we'd
    # need to know the include dir already
    set(_boost_BOOSTIFIED_VERSION)

    # Transform 1.35 => 1_35 and 1.36.0 => 1_36_0
    if(_boost_VER MATCHES "([0-9]+)\\.([0-9]+)\\.([0-9]+)")
        set(_boost_BOOSTIFIED_VERSION
          "${CMAKE_MATCH_1}_${CMAKE_MATCH_2}_${CMAKE_MATCH_3}")
    elseif(_boost_VER MATCHES "([0-9]+)\\.([0-9]+)")
        set(_boost_BOOSTIFIED_VERSION
          "${CMAKE_MATCH_1}_${CMAKE_MATCH_2}")
    endif()

    list(APPEND _boost_PATH_SUFFIXES
      "boost-${_boost_BOOSTIFIED_VERSION}"
      "boost_${_boost_BOOSTIFIED_VERSION}"
      "boost/boost-${_boost_BOOSTIFIED_VERSION}"
      "boost/boost_${_boost_BOOSTIFIED_VERSION}"
      )

  endforeach()

  _Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_boost_INCLUDE_SEARCH_DIRS")
  _Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_boost_PATH_SUFFIXES")

  # Look for a standard boost header file.
  find_path(Boost_INCLUDE_DIR
    NAMES         boost/config.hpp
    HINTS         ${_boost_INCLUDE_SEARCH_DIRS}
    PATH_SUFFIXES ${_boost_PATH_SUFFIXES}
    )
endif()

# ------------------------------------------------------------------------
#  Extract version information from version.hpp
# ------------------------------------------------------------------------

if(Boost_INCLUDE_DIR)
  _Boost_DEBUG_PRINT("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                     "location of version.hpp: ${Boost_INCLUDE_DIR}/boost/version.hpp")

  # Extract Boost_VERSION_MACRO and Boost_LIB_VERSION from version.hpp
  set(Boost_VERSION_MACRO 0)
  set(Boost_LIB_VERSION "")
  file(STRINGS "${Boost_INCLUDE_DIR}/boost/version.hpp" _boost_VERSION_HPP_CONTENTS REGEX "#define BOOST_(LIB_)?VERSION ")
  if("${_boost_VERSION_HPP_CONTENTS}" MATCHES "#define BOOST_VERSION ([0-9]+)")
    set(Boost_VERSION_MACRO "${CMAKE_MATCH_1}")
  endif()
  if("${_boost_VERSION_HPP_CONTENTS}" MATCHES "#define BOOST_LIB_VERSION \"([0-9_]+)\"")
    set(Boost_LIB_VERSION "${CMAKE_MATCH_1}")
  endif()
  unset(_boost_VERSION_HPP_CONTENTS)

  # Calculate version components
  math(EXPR Boost_VERSION_MAJOR "${Boost_VERSION_MACRO} / 100000")
  math(EXPR Boost_VERSION_MINOR "${Boost_VERSION_MACRO} / 100 % 1000")
  math(EXPR Boost_VERSION_PATCH "${Boost_VERSION_MACRO} % 100")
  set(Boost_VERSION_COUNT 3)

  # Define alias variables for backwards compat.
  set(Boost_MAJOR_VERSION ${Boost_VERSION_MAJOR})
  set(Boost_MINOR_VERSION ${Boost_VERSION_MINOR})
  set(Boost_SUBMINOR_VERSION ${Boost_VERSION_PATCH})

  # Define Boost version in x.y.z format
  set(Boost_VERSION_STRING "${Boost_VERSION_MAJOR}.${Boost_VERSION_MINOR}.${Boost_VERSION_PATCH}")

  # Define final Boost_VERSION
  cmake_policy(GET CMP0093 _Boost_CMP0093
    PARENT_SCOPE # undocumented, do not use outside of CMake
  )
  if("x${_Boost_CMP0093}x" STREQUAL "xNEWx")
    set(Boost_VERSION ${Boost_VERSION_STRING})
  else()
    set(Boost_VERSION ${Boost_VERSION_MACRO})
  endif()
  unset(_Boost_CMP0093)

  _Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "Boost_VERSION")
  _Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "Boost_VERSION_STRING")
  _Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "Boost_VERSION_MACRO")
  _Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "Boost_VERSION_MAJOR")
  _Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "Boost_VERSION_MINOR")
  _Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "Boost_VERSION_PATCH")
  _Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "Boost_VERSION_COUNT")
endif()

# ------------------------------------------------------------------------
#  Prefix initialization
# ------------------------------------------------------------------------

if ( NOT DEFINED Boost_LIB_PREFIX )
  # Boost's static libraries use a "lib" prefix on DLL platforms
  # to distinguish them from the DLL import libraries.
  if (Boost_USE_STATIC_LIBS AND (
      (WIN32 AND NOT CYGWIN)
      OR GHSMULTI
      ))
    set(Boost_LIB_PREFIX "lib")
  else()
    set(Boost_LIB_PREFIX "")
  endif()
endif()

if ( NOT Boost_NAMESPACE )
  set(Boost_NAMESPACE "boost")
endif()

_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "Boost_LIB_PREFIX")
_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "Boost_NAMESPACE")

# ------------------------------------------------------------------------
#  Suffix initialization and compiler suffix detection.
# ------------------------------------------------------------------------

set(_Boost_VARS_NAME
  Boost_NAMESPACE
  Boost_COMPILER
  Boost_THREADAPI
  Boost_USE_DEBUG_PYTHON
  Boost_USE_MULTITHREADED
  Boost_USE_STATIC_LIBS
  Boost_USE_STATIC_RUNTIME
  Boost_USE_STLPORT
  Boost_USE_STLPORT_DEPRECATED_NATIVE_IOSTREAMS
  )
_Boost_CHANGE_DETECT(_Boost_CHANGE_LIBNAME ${_Boost_VARS_NAME})

# Setting some more suffixes for the library
if (Boost_COMPILER)
  set(_boost_COMPILER ${Boost_COMPILER})
  _Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                         "_boost_COMPILER" SOURCE "user-specified via Boost_COMPILER")
else()
  # Attempt to guess the compiler suffix
  # NOTE: this is not perfect yet, if you experience any issues
  # please report them and use the Boost_COMPILER variable
  # to work around the problems.
  _Boost_GUESS_COMPILER_PREFIX(_boost_COMPILER)
endif()

set (_boost_MULTITHREADED "-mt")
if( NOT Boost_USE_MULTITHREADED )
  set (_boost_MULTITHREADED "")
endif()
_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_boost_MULTITHREADED")

#======================
# Systematically build up the Boost ABI tag for the 'tagged' and 'versioned' layouts
# http://boost.org/doc/libs/1_66_0/more/getting_started/windows.html#library-naming
# http://boost.org/doc/libs/1_66_0/boost/config/auto_link.hpp
# http://boost.org/doc/libs/1_66_0/tools/build/src/tools/common.jam
# http://boost.org/doc/libs/1_66_0/boostcpp.jam
set( _boost_RELEASE_ABI_TAG "-")
set( _boost_DEBUG_ABI_TAG   "-")
# Key       Use this library when:
#  s        linking statically to the C++ standard library and
#           compiler runtime support libraries.
if(Boost_USE_STATIC_RUNTIME)
  set( _boost_RELEASE_ABI_TAG "${_boost_RELEASE_ABI_TAG}s")
  set( _boost_DEBUG_ABI_TAG   "${_boost_DEBUG_ABI_TAG}s")
endif()
#  g        using debug versions of the standard and runtime
#           support libraries
if(WIN32 AND Boost_USE_DEBUG_RUNTIME)
  if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC"
          OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang"
          OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xIntel"
          OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xIntelLLVM")
    string(APPEND _boost_DEBUG_ABI_TAG "g")
  endif()
endif()
#  y        using special debug build of python
if(Boost_USE_DEBUG_PYTHON)
  string(APPEND _boost_DEBUG_ABI_TAG "y")
endif()
#  d        using a debug version of your code
string(APPEND _boost_DEBUG_ABI_TAG "d")
#  p        using the STLport standard library rather than the
#           default one supplied with your compiler
if(Boost_USE_STLPORT)
  string(APPEND _boost_RELEASE_ABI_TAG "p")
  string(APPEND _boost_DEBUG_ABI_TAG "p")
endif()
#  n        using the STLport deprecated "native iostreams" feature
#           removed from the documentation in 1.43.0 but still present in
#           boost/config/auto_link.hpp
if(Boost_USE_STLPORT_DEPRECATED_NATIVE_IOSTREAMS)
  string(APPEND _boost_RELEASE_ABI_TAG "n")
  string(APPEND _boost_DEBUG_ABI_TAG "n")
endif()

#  -x86     Architecture and address model tag
#           First character is the architecture, then word-size, either 32 or 64
#           Only used in 'versioned' layout, added in Boost 1.66.0
if(DEFINED Boost_ARCHITECTURE)
  set(_boost_ARCHITECTURE_TAG "${Boost_ARCHITECTURE}")
  _Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                         "_boost_ARCHITECTURE_TAG" SOURCE "user-specified via Boost_ARCHITECTURE")
else()
  set(_boost_ARCHITECTURE_TAG "")
  # {CMAKE_CXX_COMPILER_ARCHITECTURE_ID} is not currently set for all compilers
  if(NOT "x${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" STREQUAL "x" AND NOT Boost_VERSION_STRING VERSION_LESS 1.66.0)
    string(APPEND _boost_ARCHITECTURE_TAG "-")
    # This needs to be kept in-sync with the section of CMakePlatformId.h.in
    # inside 'defined(_WIN32) && defined(_MSC_VER)'
    if(CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "IA64")
      string(APPEND _boost_ARCHITECTURE_TAG "i")
    elseif(CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "X86"
              OR CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64")
      string(APPEND _boost_ARCHITECTURE_TAG "x")
    elseif(CMAKE_CXX_COMPILER_ARCHITECTURE_ID MATCHES "^ARM")
      string(APPEND _boost_ARCHITECTURE_TAG "a")
    elseif(CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "MIPS")
      string(APPEND _boost_ARCHITECTURE_TAG "m")
    endif()

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      string(APPEND _boost_ARCHITECTURE_TAG "64")
    else()
      string(APPEND _boost_ARCHITECTURE_TAG "32")
    endif()
  endif()
  _Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                         "_boost_ARCHITECTURE_TAG" SOURCE "detected")
endif()

_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_boost_RELEASE_ABI_TAG")
_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_boost_DEBUG_ABI_TAG")

# ------------------------------------------------------------------------
#  Begin finding boost libraries
# ------------------------------------------------------------------------

set(_Boost_VARS_LIB "")
foreach(c DEBUG RELEASE)
  set(_Boost_VARS_LIB_${c} BOOST_LIBRARYDIR Boost_LIBRARY_DIR_${c})
  list(APPEND _Boost_VARS_LIB ${_Boost_VARS_LIB_${c}})
  _Boost_CHANGE_DETECT(_Boost_CHANGE_LIBDIR_${c} ${_Boost_VARS_DIR} ${_Boost_VARS_LIB_${c}} Boost_INCLUDE_DIR)
  # Clear Boost_LIBRARY_DIR_${c} if it did not change but other input affecting the
  # location did.  We will find a new one based on the new inputs.
  if(_Boost_CHANGE_LIBDIR_${c} AND NOT _Boost_LIBRARY_DIR_${c}_CHANGED)
    unset(Boost_LIBRARY_DIR_${c} CACHE)
  endif()

  # If Boost_LIBRARY_DIR_[RELEASE,DEBUG] is set, prefer its value.
  if(Boost_LIBRARY_DIR_${c})
    set(_boost_LIBRARY_SEARCH_DIRS_${c} ${Boost_LIBRARY_DIR_${c}} NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  else()
    set(_boost_LIBRARY_SEARCH_DIRS_${c} "")
    if(BOOST_LIBRARYDIR)
      list(APPEND _boost_LIBRARY_SEARCH_DIRS_${c} ${BOOST_LIBRARYDIR})
    elseif(_ENV_BOOST_LIBRARYDIR)
      list(APPEND _boost_LIBRARY_SEARCH_DIRS_${c} ${_ENV_BOOST_LIBRARYDIR})
    endif()

    if(BOOST_ROOT)
      list(APPEND _boost_LIBRARY_SEARCH_DIRS_${c} ${BOOST_ROOT}/lib ${BOOST_ROOT}/stage/lib)
      _Boost_UPDATE_WINDOWS_LIBRARY_SEARCH_DIRS_WITH_PREBUILT_PATHS(_boost_LIBRARY_SEARCH_DIRS_${c} "${BOOST_ROOT}")
    elseif(_ENV_BOOST_ROOT)
      list(APPEND _boost_LIBRARY_SEARCH_DIRS_${c} ${_ENV_BOOST_ROOT}/lib ${_ENV_BOOST_ROOT}/stage/lib)
      _Boost_UPDATE_WINDOWS_LIBRARY_SEARCH_DIRS_WITH_PREBUILT_PATHS(_boost_LIBRARY_SEARCH_DIRS_${c} "${_ENV_BOOST_ROOT}")
    endif()

    list(APPEND _boost_LIBRARY_SEARCH_DIRS_${c}
      ${Boost_INCLUDE_DIR}/lib
      ${Boost_INCLUDE_DIR}/../lib
      ${Boost_INCLUDE_DIR}/stage/lib
      )
    _Boost_UPDATE_WINDOWS_LIBRARY_SEARCH_DIRS_WITH_PREBUILT_PATHS(_boost_LIBRARY_SEARCH_DIRS_${c} "${Boost_INCLUDE_DIR}/..")
    _Boost_UPDATE_WINDOWS_LIBRARY_SEARCH_DIRS_WITH_PREBUILT_PATHS(_boost_LIBRARY_SEARCH_DIRS_${c} "${Boost_INCLUDE_DIR}")
    if( Boost_NO_SYSTEM_PATHS )
      list(APPEND _boost_LIBRARY_SEARCH_DIRS_${c} NO_CMAKE_SYSTEM_PATH NO_SYSTEM_ENVIRONMENT_PATH)
    else()
      foreach(ver ${_boost_TEST_VERSIONS})
        string(REPLACE "." "_" ver "${ver}")
        _Boost_UPDATE_WINDOWS_LIBRARY_SEARCH_DIRS_WITH_PREBUILT_PATHS(_boost_LIBRARY_SEARCH_DIRS_${c} "C:/local/boost_${ver}")
      endforeach()
      _Boost_UPDATE_WINDOWS_LIBRARY_SEARCH_DIRS_WITH_PREBUILT_PATHS(_boost_LIBRARY_SEARCH_DIRS_${c} "C:/boost")
      list(APPEND _boost_LIBRARY_SEARCH_DIRS_${c} PATHS
        C:/boost/lib
        C:/boost
        /sw/local/lib
        )
    endif()
  endif()
endforeach()

_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_boost_LIBRARY_SEARCH_DIRS_RELEASE")
_Boost_DEBUG_PRINT_VAR("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}" "_boost_LIBRARY_SEARCH_DIRS_DEBUG")

# Support preference of static libs by adjusting CMAKE_FIND_LIBRARY_SUFFIXES
if( Boost_USE_STATIC_LIBS )
  set( _boost_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  if(WIN32)
    list(INSERT CMAKE_FIND_LIBRARY_SUFFIXES 0 .lib .a)
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
  endif()
endif()

# We want to use the tag inline below without risking double dashes
if(_boost_RELEASE_ABI_TAG)
  if(${_boost_RELEASE_ABI_TAG} STREQUAL "-")
    set(_boost_RELEASE_ABI_TAG "")
  endif()
endif()
if(_boost_DEBUG_ABI_TAG)
  if(${_boost_DEBUG_ABI_TAG} STREQUAL "-")
    set(_boost_DEBUG_ABI_TAG "")
  endif()
endif()

# The previous behavior of FindBoost when Boost_USE_STATIC_LIBS was enabled
# on WIN32 was to:
#  1. Search for static libs compiled against a SHARED C++ standard runtime library (use if found)
#  2. Search for static libs compiled against a STATIC C++ standard runtime library (use if found)
# We maintain this behavior since changing it could break people's builds.
# To disable the ambiguous behavior, the user need only
# set Boost_USE_STATIC_RUNTIME either ON or OFF.
set(_boost_STATIC_RUNTIME_WORKAROUND false)
if(WIN32 AND Boost_USE_STATIC_LIBS)
  if(NOT DEFINED Boost_USE_STATIC_RUNTIME)
    set(_boost_STATIC_RUNTIME_WORKAROUND TRUE)
  endif()
endif()

# On versions < 1.35, remove the System library from the considered list
# since it wasn't added until 1.35.
if(Boost_VERSION_STRING AND Boost_FIND_COMPONENTS)
  if(Boost_VERSION_STRING VERSION_LESS 1.35.0)
    list(REMOVE_ITEM Boost_FIND_COMPONENTS system)
  endif()
endif()

# Additional components may be required via component dependencies.
# Add any missing components to the list.
_Boost_MISSING_DEPENDENCIES(Boost_FIND_COMPONENTS _Boost_EXTRA_FIND_COMPONENTS)

# If thread is required, get the thread libs as a dependency
if("thread" IN_LIST Boost_FIND_COMPONENTS)
  if(Boost_FIND_QUIETLY)
    set(_Boost_find_quiet QUIET)
  else()
    set(_Boost_find_quiet "")
  endif()
  find_package(Threads ${_Boost_find_quiet})
  unset(_Boost_find_quiet)
endif()

# If the user changed any of our control inputs flush previous results.
if(_Boost_CHANGE_LIBDIR_DEBUG OR _Boost_CHANGE_LIBDIR_RELEASE OR _Boost_CHANGE_LIBNAME)
  foreach(COMPONENT ${_Boost_COMPONENTS_SEARCHED})
    string(TOUPPER ${COMPONENT} UPPERCOMPONENT)
    foreach(c DEBUG RELEASE)
      set(_var Boost_${UPPERCOMPONENT}_LIBRARY_${c})
      unset(${_var} CACHE)
      set(${_var} "${_var}-NOTFOUND")
    endforeach()
  endforeach()
  set(_Boost_COMPONENTS_SEARCHED "")
endif()

foreach(COMPONENT ${Boost_FIND_COMPONENTS})
  string(TOUPPER ${COMPONENT} UPPERCOMPONENT)

  set( _boost_docstring_release "Boost ${COMPONENT} library (release)")
  set( _boost_docstring_debug   "Boost ${COMPONENT} library (debug)")

  # Compute component-specific hints.
  set(_Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT "")
  if(${COMPONENT} STREQUAL "mpi" OR ${COMPONENT} STREQUAL "mpi_python" OR
     ${COMPONENT} STREQUAL "graph_parallel")
    foreach(lib ${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES})
      if(IS_ABSOLUTE "${lib}")
        get_filename_component(libdir "${lib}" PATH)
        string(REPLACE "\\" "/" libdir "${libdir}")
        list(APPEND _Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT ${libdir})
      endif()
    endforeach()
  endif()

  # Handle Python version suffixes
  unset(COMPONENT_PYTHON_VERSION_MAJOR)
  unset(COMPONENT_PYTHON_VERSION_MINOR)
  if(${COMPONENT} MATCHES "^(python|mpi_python|numpy)([0-9])\$")
    set(COMPONENT_UNVERSIONED "${CMAKE_MATCH_1}")
    set(COMPONENT_PYTHON_VERSION_MAJOR "${CMAKE_MATCH_2}")
  elseif(${COMPONENT} MATCHES "^(python|mpi_python|numpy)([0-9])\\.?([0-9]+)\$")
    set(COMPONENT_UNVERSIONED "${CMAKE_MATCH_1}")
    set(COMPONENT_PYTHON_VERSION_MAJOR "${CMAKE_MATCH_2}")
    set(COMPONENT_PYTHON_VERSION_MINOR "${CMAKE_MATCH_3}")
  endif()

  unset(_Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT_NAME)
  if (COMPONENT_PYTHON_VERSION_MINOR)
    # Boost >= 1.67
    list(APPEND _Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT_NAME "${COMPONENT_UNVERSIONED}${COMPONENT_PYTHON_VERSION_MAJOR}${COMPONENT_PYTHON_VERSION_MINOR}")
    # Debian/Ubuntu (Some versions omit the 2 and/or 3 from the suffix)
    list(APPEND _Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT_NAME "${COMPONENT_UNVERSIONED}${COMPONENT_PYTHON_VERSION_MAJOR}-py${COMPONENT_PYTHON_VERSION_MAJOR}${COMPONENT_PYTHON_VERSION_MINOR}")
    list(APPEND _Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT_NAME "${COMPONENT_UNVERSIONED}-py${COMPONENT_PYTHON_VERSION_MAJOR}${COMPONENT_PYTHON_VERSION_MINOR}")
    # Gentoo
    list(APPEND _Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT_NAME "${COMPONENT_UNVERSIONED}-${COMPONENT_PYTHON_VERSION_MAJOR}.${COMPONENT_PYTHON_VERSION_MINOR}")
    # RPMs
    list(APPEND _Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT_NAME "${COMPONENT_UNVERSIONED}-${COMPONENT_PYTHON_VERSION_MAJOR}${COMPONENT_PYTHON_VERSION_MINOR}")
  endif()
  if (COMPONENT_PYTHON_VERSION_MAJOR AND NOT COMPONENT_PYTHON_VERSION_MINOR)
    # Boost < 1.67
    list(APPEND _Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT_NAME "${COMPONENT_UNVERSIONED}${COMPONENT_PYTHON_VERSION_MAJOR}")
  endif()

  # Consolidate and report component-specific hints.
  if(_Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT_NAME)
    list(REMOVE_DUPLICATES _Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT_NAME)
    _Boost_DEBUG_PRINT("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
      "Component-specific library search names for ${COMPONENT_NAME}: ${_Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT_NAME}")
  endif()
  if(_Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT)
    list(REMOVE_DUPLICATES _Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT)
    _Boost_DEBUG_PRINT("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
      "Component-specific library search paths for ${COMPONENT}: ${_Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT}")
  endif()

  #
  # Find headers
  #
  _Boost_COMPONENT_HEADERS("${COMPONENT}" Boost_${UPPERCOMPONENT}_HEADER_NAME)
  # Look for a standard boost header file.
  if(Boost_${UPPERCOMPONENT}_HEADER_NAME)
    if(EXISTS "${Boost_INCLUDE_DIR}/${Boost_${UPPERCOMPONENT}_HEADER_NAME}")
      set(Boost_${UPPERCOMPONENT}_HEADER ON)
    else()
      set(Boost_${UPPERCOMPONENT}_HEADER OFF)
    endif()
  else()
    set(Boost_${UPPERCOMPONENT}_HEADER ON)
    message(WARNING "No header defined for ${COMPONENT}; skipping header check "
                    "(note: header-only libraries have no designated component)")
  endif()

  #
  # Find RELEASE libraries
  #
  unset(_boost_RELEASE_NAMES)
  foreach(component IN LISTS _Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT_NAME COMPONENT)
    foreach(compiler IN LISTS _boost_COMPILER)
      list(APPEND _boost_RELEASE_NAMES
        ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${compiler}${_boost_MULTITHREADED}${_boost_RELEASE_ABI_TAG}${_boost_ARCHITECTURE_TAG}-${Boost_LIB_VERSION}
        ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${compiler}${_boost_MULTITHREADED}${_boost_RELEASE_ABI_TAG}${_boost_ARCHITECTURE_TAG}
        ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${compiler}${_boost_MULTITHREADED}${_boost_RELEASE_ABI_TAG} )
    endforeach()
    list(APPEND _boost_RELEASE_NAMES
      ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${_boost_MULTITHREADED}${_boost_RELEASE_ABI_TAG}${_boost_ARCHITECTURE_TAG}-${Boost_LIB_VERSION}
      ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${_boost_MULTITHREADED}${_boost_RELEASE_ABI_TAG}${_boost_ARCHITECTURE_TAG}
      ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${_boost_MULTITHREADED}${_boost_RELEASE_ABI_TAG}
      ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${_boost_MULTITHREADED}
      ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component} )
    if(_boost_STATIC_RUNTIME_WORKAROUND)
      set(_boost_RELEASE_STATIC_ABI_TAG "-s${_boost_RELEASE_ABI_TAG}")
      foreach(compiler IN LISTS _boost_COMPILER)
        list(APPEND _boost_RELEASE_NAMES
          ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${compiler}${_boost_MULTITHREADED}${_boost_RELEASE_STATIC_ABI_TAG}${_boost_ARCHITECTURE_TAG}-${Boost_LIB_VERSION}
          ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${compiler}${_boost_MULTITHREADED}${_boost_RELEASE_STATIC_ABI_TAG}${_boost_ARCHITECTURE_TAG}
          ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${compiler}${_boost_MULTITHREADED}${_boost_RELEASE_STATIC_ABI_TAG} )
      endforeach()
      list(APPEND _boost_RELEASE_NAMES
        ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${_boost_MULTITHREADED}${_boost_RELEASE_STATIC_ABI_TAG}${_boost_ARCHITECTURE_TAG}-${Boost_LIB_VERSION}
        ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${_boost_MULTITHREADED}${_boost_RELEASE_STATIC_ABI_TAG}${_boost_ARCHITECTURE_TAG}
        ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${_boost_MULTITHREADED}${_boost_RELEASE_STATIC_ABI_TAG} )
    endif()
  endforeach()
  if(Boost_THREADAPI AND ${COMPONENT} STREQUAL "thread")
    _Boost_PREPEND_LIST_WITH_THREADAPI(_boost_RELEASE_NAMES ${_boost_RELEASE_NAMES})
  endif()
  _Boost_DEBUG_PRINT("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                     "Searching for ${UPPERCOMPONENT}_LIBRARY_RELEASE: ${_boost_RELEASE_NAMES}")

  # if Boost_LIBRARY_DIR_RELEASE is not defined,
  # but Boost_LIBRARY_DIR_DEBUG is, look there first for RELEASE libs
  if(NOT Boost_LIBRARY_DIR_RELEASE AND Boost_LIBRARY_DIR_DEBUG)
    list(INSERT _boost_LIBRARY_SEARCH_DIRS_RELEASE 0 ${Boost_LIBRARY_DIR_DEBUG})
  endif()

  # Avoid passing backslashes to _Boost_FIND_LIBRARY due to macro re-parsing.
  string(REPLACE "\\" "/" _boost_LIBRARY_SEARCH_DIRS_tmp "${_boost_LIBRARY_SEARCH_DIRS_RELEASE}")

  if(Boost_USE_RELEASE_LIBS)
    _Boost_FIND_LIBRARY(Boost_${UPPERCOMPONENT}_LIBRARY_RELEASE RELEASE
      NAMES ${_boost_RELEASE_NAMES}
      HINTS ${_boost_LIBRARY_SEARCH_DIRS_tmp}
      NAMES_PER_DIR
      DOC "${_boost_docstring_release}"
      )
  endif()

  #
  # Find DEBUG libraries
  #
  unset(_boost_DEBUG_NAMES)
  foreach(component IN LISTS _Boost_FIND_LIBRARY_HINTS_FOR_COMPONENT_NAME COMPONENT)
    foreach(compiler IN LISTS _boost_COMPILER)
      list(APPEND _boost_DEBUG_NAMES
        ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${compiler}${_boost_MULTITHREADED}${_boost_DEBUG_ABI_TAG}${_boost_ARCHITECTURE_TAG}-${Boost_LIB_VERSION}
        ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${compiler}${_boost_MULTITHREADED}${_boost_DEBUG_ABI_TAG}${_boost_ARCHITECTURE_TAG}
        ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${compiler}${_boost_MULTITHREADED}${_boost_DEBUG_ABI_TAG} )
    endforeach()
    list(APPEND _boost_DEBUG_NAMES
      ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${_boost_MULTITHREADED}${_boost_DEBUG_ABI_TAG}${_boost_ARCHITECTURE_TAG}-${Boost_LIB_VERSION}
      ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${_boost_MULTITHREADED}${_boost_DEBUG_ABI_TAG}${_boost_ARCHITECTURE_TAG}
      ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${_boost_MULTITHREADED}${_boost_DEBUG_ABI_TAG}
      ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${_boost_MULTITHREADED}
      ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component} )
    if(_boost_STATIC_RUNTIME_WORKAROUND)
      set(_boost_DEBUG_STATIC_ABI_TAG "-s${_boost_DEBUG_ABI_TAG}")
      foreach(compiler IN LISTS _boost_COMPILER)
        list(APPEND _boost_DEBUG_NAMES
          ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${compiler}${_boost_MULTITHREADED}${_boost_DEBUG_STATIC_ABI_TAG}${_boost_ARCHITECTURE_TAG}-${Boost_LIB_VERSION}
          ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${compiler}${_boost_MULTITHREADED}${_boost_DEBUG_STATIC_ABI_TAG}${_boost_ARCHITECTURE_TAG}
          ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${compiler}${_boost_MULTITHREADED}${_boost_DEBUG_STATIC_ABI_TAG} )
      endforeach()
      list(APPEND _boost_DEBUG_NAMES
        ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${_boost_MULTITHREADED}${_boost_DEBUG_STATIC_ABI_TAG}${_boost_ARCHITECTURE_TAG}-${Boost_LIB_VERSION}
        ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${_boost_MULTITHREADED}${_boost_DEBUG_STATIC_ABI_TAG}${_boost_ARCHITECTURE_TAG}
        ${Boost_LIB_PREFIX}${Boost_NAMESPACE}_${component}${_boost_MULTITHREADED}${_boost_DEBUG_STATIC_ABI_TAG} )
    endif()
  endforeach()
  if(Boost_THREADAPI AND ${COMPONENT} STREQUAL "thread")
     _Boost_PREPEND_LIST_WITH_THREADAPI(_boost_DEBUG_NAMES ${_boost_DEBUG_NAMES})
  endif()
  _Boost_DEBUG_PRINT("${CMAKE_CURRENT_LIST_FILE}" "${CMAKE_CURRENT_LIST_LINE}"
                     "Searching for ${UPPERCOMPONENT}_LIBRARY_DEBUG: ${_boost_DEBUG_NAMES}")

  # if Boost_LIBRARY_DIR_DEBUG is not defined,
  # but Boost_LIBRARY_DIR_RELEASE is, look there first for DEBUG libs
  if(NOT Boost_LIBRARY_DIR_DEBUG AND Boost_LIBRARY_DIR_RELEASE)
    list(INSERT _boost_LIBRARY_SEARCH_DIRS_DEBUG 0 ${Boost_LIBRARY_DIR_RELEASE})
  endif()

  # Avoid passing backslashes to _Boost_FIND_LIBRARY due to macro re-parsing.
  string(REPLACE "\\" "/" _boost_LIBRARY_SEARCH_DIRS_tmp "${_boost_LIBRARY_SEARCH_DIRS_DEBUG}")

  if(Boost_USE_DEBUG_LIBS)
    _Boost_FIND_LIBRARY(Boost_${UPPERCOMPONENT}_LIBRARY_DEBUG DEBUG
      NAMES ${_boost_DEBUG_NAMES}
      HINTS ${_boost_LIBRARY_SEARCH_DIRS_tmp}
      NAMES_PER_DIR
      DOC "${_boost_docstring_debug}"
      )
  endif ()

  if(Boost_REALPATH)
    _Boost_SWAP_WITH_REALPATH(Boost_${UPPERCOMPONENT}_LIBRARY_RELEASE "${_boost_docstring_release}")
    _Boost_SWAP_WITH_REALPATH(Boost_${UPPERCOMPONENT}_LIBRARY_DEBUG   "${_boost_docstring_debug}"  )
  endif()

  _Boost_ADJUST_LIB_VARS(${UPPERCOMPONENT})

  # Check if component requires some compiler features
  _Boost_COMPILER_FEATURES(${COMPONENT} _Boost_${UPPERCOMPONENT}_COMPILER_FEATURES)

endforeach()

# Restore the original find library ordering
if( Boost_USE_STATIC_LIBS )
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_boost_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

# ------------------------------------------------------------------------
#  End finding boost libraries
# ------------------------------------------------------------------------

set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIR})
set(Boost_LIBRARY_DIRS)
if(Boost_LIBRARY_DIR_RELEASE)
  list(APPEND Boost_LIBRARY_DIRS ${Boost_LIBRARY_DIR_RELEASE})
endif()
if(Boost_LIBRARY_DIR_DEBUG)
  list(APPEND Boost_LIBRARY_DIRS ${Boost_LIBRARY_DIR_DEBUG})
endif()
if(Boost_LIBRARY_DIRS)
  list(REMOVE_DUPLICATES Boost_LIBRARY_DIRS)
endif()

# ------------------------------------------------------------------------
#  Call FPHSA helper, see https://cmake.org/cmake/help/latest/module/FindPackageHandleStandardArgs.html
# ------------------------------------------------------------------------

# Define aliases as needed by the component handler in the FPHSA helper below
foreach(_comp IN LISTS Boost_FIND_COMPONENTS)
  string(TOUPPER ${_comp} _uppercomp)
  if(DEFINED Boost_${_uppercomp}_FOUND)
    set(Boost_${_comp}_FOUND ${Boost_${_uppercomp}_FOUND})
  endif()
endforeach()

find_package_handle_standard_args(Boost
  REQUIRED_VARS Boost_INCLUDE_DIR
  VERSION_VAR Boost_VERSION_STRING
  HANDLE_COMPONENTS)

if(Boost_FOUND)
  if( NOT Boost_LIBRARY_DIRS )
    # Compatibility Code for backwards compatibility with CMake
    # 2.4's FindBoost module.

    # Look for the boost library path.
    # Note that the user may not have installed any libraries
    # so it is quite possible the Boost_LIBRARY_DIRS may not exist.
    set(_boost_LIB_DIR ${Boost_INCLUDE_DIR})

    if("${_boost_LIB_DIR}" MATCHES "boost-[0-9]+")
      get_filename_component(_boost_LIB_DIR ${_boost_LIB_DIR} PATH)
    endif()

    if("${_boost_LIB_DIR}" MATCHES "/include$")
      # Strip off the trailing "/include" in the path.
      get_filename_component(_boost_LIB_DIR ${_boost_LIB_DIR} PATH)
    endif()

    if(EXISTS "${_boost_LIB_DIR}/lib")
      string(APPEND _boost_LIB_DIR /lib)
    elseif(EXISTS "${_boost_LIB_DIR}/stage/lib")
      string(APPEND _boost_LIB_DIR "/stage/lib")
    else()
      set(_boost_LIB_DIR "")
    endif()

    if(_boost_LIB_DIR AND EXISTS "${_boost_LIB_DIR}")
      set(Boost_LIBRARY_DIRS ${_boost_LIB_DIR})
    endif()

  endif()
else()
  # Boost headers were not found so no components were found.
  foreach(COMPONENT ${Boost_FIND_COMPONENTS})
    string(TOUPPER ${COMPONENT} UPPERCOMPONENT)
    set(Boost_${UPPERCOMPONENT}_FOUND 0)
  endforeach()
endif()

# ------------------------------------------------------------------------
#  Add imported targets
# ------------------------------------------------------------------------

if(Boost_FOUND)
  # The builtin CMake package in Boost 1.70+ introduces a new name
  # for the header-only lib, let's provide the same UI in module mode
  if(NOT TARGET Boost::headers)
    add_library(Boost::headers INTERFACE IMPORTED)
    if(Boost_INCLUDE_DIRS)
      set_target_properties(Boost::headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
    endif()
  endif()

  # Define the old target name for header-only libraries for backwards
  # compat.
  if(NOT TARGET Boost::boost)
    add_library(Boost::boost INTERFACE IMPORTED)
    set_target_properties(Boost::boost
      PROPERTIES INTERFACE_LINK_LIBRARIES Boost::headers)
  endif()

  foreach(COMPONENT ${Boost_FIND_COMPONENTS})
    if(_Boost_IMPORTED_TARGETS AND NOT TARGET Boost::${COMPONENT})
      string(TOUPPER ${COMPONENT} UPPERCOMPONENT)
      if(Boost_${UPPERCOMPONENT}_FOUND)
        if(Boost_USE_STATIC_LIBS)
          add_library(Boost::${COMPONENT} STATIC IMPORTED)
        else()
          # Even if Boost_USE_STATIC_LIBS is OFF, we might have static
          # libraries as a result.
          add_library(Boost::${COMPONENT} UNKNOWN IMPORTED)
        endif()
        if(Boost_INCLUDE_DIRS)
          set_target_properties(Boost::${COMPONENT} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
        endif()
        if(EXISTS "${Boost_${UPPERCOMPONENT}_LIBRARY}")
          set_target_properties(Boost::${COMPONENT} PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
            IMPORTED_LOCATION "${Boost_${UPPERCOMPONENT}_LIBRARY}")
        endif()
        if(EXISTS "${Boost_${UPPERCOMPONENT}_LIBRARY_RELEASE}")
          set_property(TARGET Boost::${COMPONENT} APPEND PROPERTY
            IMPORTED_CONFIGURATIONS RELEASE)
          set_target_properties(Boost::${COMPONENT} PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
            IMPORTED_LOCATION_RELEASE "${Boost_${UPPERCOMPONENT}_LIBRARY_RELEASE}")
        endif()
        if(EXISTS "${Boost_${UPPERCOMPONENT}_LIBRARY_DEBUG}")
          set_property(TARGET Boost::${COMPONENT} APPEND PROPERTY
            IMPORTED_CONFIGURATIONS DEBUG)
          set_target_properties(Boost::${COMPONENT} PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
            IMPORTED_LOCATION_DEBUG "${Boost_${UPPERCOMPONENT}_LIBRARY_DEBUG}")
        endif()
        if(_Boost_${UPPERCOMPONENT}_DEPENDENCIES)
          unset(_Boost_${UPPERCOMPONENT}_TARGET_DEPENDENCIES)
          foreach(dep ${_Boost_${UPPERCOMPONENT}_DEPENDENCIES})
            list(APPEND _Boost_${UPPERCOMPONENT}_TARGET_DEPENDENCIES Boost::${dep})
          endforeach()
          if(COMPONENT STREQUAL "thread")
            list(APPEND _Boost_${UPPERCOMPONENT}_TARGET_DEPENDENCIES Threads::Threads)
          endif()
          set_target_properties(Boost::${COMPONENT} PROPERTIES
            INTERFACE_LINK_LIBRARIES "${_Boost_${UPPERCOMPONENT}_TARGET_DEPENDENCIES}")
        endif()
        if(_Boost_${UPPERCOMPONENT}_COMPILER_FEATURES)
          set_target_properties(Boost::${COMPONENT} PROPERTIES
            INTERFACE_COMPILE_FEATURES "${_Boost_${UPPERCOMPONENT}_COMPILER_FEATURES}")
        endif()
      endif()
    endif()
  endforeach()

  # Supply Boost_LIB_DIAGNOSTIC_DEFINITIONS as a convenience target. It
  # will only contain any interface definitions on WIN32, but is created
  # on all platforms to keep end user code free from platform dependent
  # code.  Also provide convenience targets to disable autolinking and
  # enable dynamic linking.
  if(NOT TARGET Boost::diagnostic_definitions)
    add_library(Boost::diagnostic_definitions INTERFACE IMPORTED)
    add_library(Boost::disable_autolinking INTERFACE IMPORTED)
    add_library(Boost::dynamic_linking INTERFACE IMPORTED)
    set_target_properties(Boost::dynamic_linking PROPERTIES
      INTERFACE_COMPILE_DEFINITIONS "BOOST_ALL_DYN_LINK")
  endif()
  if(WIN32)
    # In windows, automatic linking is performed, so you do not have
    # to specify the libraries.  If you are linking to a dynamic
    # runtime, then you can choose to link to either a static or a
    # dynamic Boost library, the default is to do a static link.  You
    # can alter this for a specific library "whatever" by defining
    # BOOST_WHATEVER_DYN_LINK to force Boost library "whatever" to be
    # linked dynamically.  Alternatively you can force all Boost
    # libraries to dynamic link by defining BOOST_ALL_DYN_LINK.

    # This feature can be disabled for Boost library "whatever" by
    # defining BOOST_WHATEVER_NO_LIB, or for all of Boost by defining
    # BOOST_ALL_NO_LIB.

    # If you want to observe which libraries are being linked against
    # then defining BOOST_LIB_DIAGNOSTIC will cause the auto-linking
    # code to emit a #pragma message each time a library is selected
    # for linking.
    set(Boost_LIB_DIAGNOSTIC_DEFINITIONS "-DBOOST_LIB_DIAGNOSTIC")
    set_target_properties(Boost::diagnostic_definitions PROPERTIES
      INTERFACE_COMPILE_DEFINITIONS "BOOST_LIB_DIAGNOSTIC")
    set_target_properties(Boost::disable_autolinking PROPERTIES
      INTERFACE_COMPILE_DEFINITIONS "BOOST_ALL_NO_LIB")
  endif()
endif()

# ------------------------------------------------------------------------
#  Finalize
# ------------------------------------------------------------------------

# Report Boost_LIBRARIES
set(Boost_LIBRARIES "")
foreach(_comp IN LISTS Boost_FIND_COMPONENTS)
  string(TOUPPER ${_comp} _uppercomp)
  if(Boost_${_uppercomp}_FOUND)
    list(APPEND Boost_LIBRARIES ${Boost_${_uppercomp}_LIBRARY})
    if(_comp STREQUAL "thread")
      list(APPEND Boost_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
    endif()
  endif()
endforeach()

# Configure display of cache entries in GUI.
foreach(v BOOSTROOT BOOST_ROOT ${_Boost_VARS_INC} ${_Boost_VARS_LIB})
  get_property(_type CACHE ${v} PROPERTY TYPE)
  if(_type)
    set_property(CACHE ${v} PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      if("x${v}" STREQUAL "xBoost_ADDITIONAL_VERSIONS")
        set_property(CACHE ${v} PROPERTY TYPE STRING)
      else()
        set_property(CACHE ${v} PROPERTY TYPE PATH)
      endif()
    endif()
  endif()
endforeach()

# Record last used values of input variables so we can
# detect on the next run if the user changed them.
foreach(v
    ${_Boost_VARS_INC} ${_Boost_VARS_LIB}
    ${_Boost_VARS_DIR} ${_Boost_VARS_NAME}
    )
  if(DEFINED ${v})
    set(_${v}_LAST "${${v}}" CACHE INTERNAL "Last used ${v} value.")
  else()
    unset(_${v}_LAST CACHE)
  endif()
endforeach()

# Maintain a persistent list of components requested anywhere since
# the last flush.
set(_Boost_COMPONENTS_SEARCHED "${_Boost_COMPONENTS_SEARCHED}")
list(APPEND _Boost_COMPONENTS_SEARCHED ${Boost_FIND_COMPONENTS})
list(REMOVE_DUPLICATES _Boost_COMPONENTS_SEARCHED)
list(SORT _Boost_COMPONENTS_SEARCHED)
set(_Boost_COMPONENTS_SEARCHED "${_Boost_COMPONENTS_SEARCHED}"
  CACHE INTERNAL "Components requested for this build tree.")

# Restore project's policies
cmake_policy(POP)
