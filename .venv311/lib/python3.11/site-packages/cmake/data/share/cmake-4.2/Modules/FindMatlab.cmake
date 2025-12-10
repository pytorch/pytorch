# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindMatlab
----------

Finds Matlab or Matlab Compiler Runtime (MCR) and provides Matlab tools,
libraries and compilers to CMake:

.. code-block:: cmake

  find_package(Matlab [<version>] [COMPONENTS <components>...] [...])

This package primary purpose is to find the libraries associated with Matlab
or the MCR in order to be able to build Matlab extensions (mex files). It
can also be used:

* to run specific commands in Matlab in case Matlab is available
* for declaring Matlab unit test
* to retrieve various information from Matlab (mex extensions, versions and
  release queries, ...)

.. versionadded:: 3.12
  Added Matlab Compiler Runtime (MCR) support.

The module supports the following components:

* ``ENG_LIBRARY`` and ``MAT_LIBRARY``: respectively the ``ENG`` and ``MAT``
  libraries of Matlab
* ``MAIN_PROGRAM`` the Matlab binary program. Note that this component is not
  available on the MCR version, and will yield an error if the MCR is found
  instead of the regular Matlab installation.
* ``MEX_COMPILER`` the MEX compiler.
* ``MCC_COMPILER`` the MCC compiler, included with the Matlab Compiler add-on.
* ``SIMULINK`` the Simulink environment.

.. versionadded:: 3.7
  Added the ``MAT_LIBRARY`` component.

.. versionadded:: 3.13
  Added the ``ENGINE_LIBRARY``, ``DATAARRAY_LIBRARY`` and ``MCC_COMPILER``
  components.

.. versionchanged:: 3.14
  Removed the ``MX_LIBRARY``, ``ENGINE_LIBRARY`` and ``DATAARRAY_LIBRARY``
  components.  These libraries are found unconditionally.

.. versionadded:: 3.30
  Added support for specifying a version range to :command:`find_package` and
  added support for specifying ``REGISTRY_VIEW`` to :command:`find_package`,
  :command:`matlab_extract_all_installed_versions_from_registry` and
  :command:`matlab_get_all_valid_matlab_roots_from_registry`. The default
  behavior remained unchanged, by using the registry view ``TARGET``.

.. note::

  The version given to the :command:`find_package` directive is the Matlab
  **version**, which should not be confused with the Matlab *release* name
  (e.g. `R2023b`).
  The :command:`matlab_get_version_from_release_name` and
  :command:`matlab_get_release_name_from_version` provide a mapping
  between the release name and the version.

The variable :variable:`Matlab_ROOT_DIR` may be specified in order to give
the path of the desired Matlab version. Otherwise, the behavior is platform
specific:

* Windows: The installed versions of Matlab/MCR are retrieved from the
  Windows registry. The ``REGISTRY_VIEW`` argument may optionally be specified
  to manually control whether 32bit or 64bit versions shall be searched for.
* macOS: The installed versions of Matlab/MCR are given by the MATLAB
  default installation paths under ``$HOME/Applications`` and ``/Applications``.
  If no such application is found, it falls back to the one that might be
  accessible from the ``PATH``.
* Unix: The desired Matlab should be accessible from the ``PATH``. This does
  not work for MCR installation and :variable:`Matlab_ROOT_DIR` should be
  specified on this platform.

Additional information is provided when :variable:`MATLAB_FIND_DEBUG` is set.
When a Matlab/MCR installation is found automatically and the ``MATLAB_VERSION``
is not given, the version is queried from Matlab directly (on Windows this
may pop up a Matlab window) or from the MCR installation.

The mapping of the release names and the version of Matlab is performed by
defining pairs (name, version).  The variable
:variable:`MATLAB_ADDITIONAL_VERSIONS` may be provided before the call to
the :command:`find_package` in order to handle additional versions.

A Matlab scripts can be added to the set of tests using the
:command:`matlab_add_unit_test`. By default, the Matlab unit test framework
will be used (>= 2013a) to run this script, but regular ``.m`` files
returning an exit code can be used as well (0 indicating a success).

Module Input Variables
^^^^^^^^^^^^^^^^^^^^^^

Users or projects may set the following variables to configure the module
behavior:

:variable:`Matlab_ROOT <<PackageName>_ROOT>`
  .. versionadded:: 3.25

  Default value for :variable:`Matlab_ROOT_DIR`, the root of the Matlab
  installation.

:variable:`Matlab_ROOT_DIR`
  The root of the Matlab installation.

:variable:`MATLAB_FIND_DEBUG`
  outputs debug information

:variable:`MATLAB_ADDITIONAL_VERSIONS`
  additional versions of Matlab for the automatic retrieval of the installed
  versions.

Imported Targets
^^^^^^^^^^^^^^^^

.. versionadded:: 3.22

This module defines the following :prop_tgt:`IMPORTED` targets:

``Matlab::mex``
  The ``mex`` library, always available for MATLAB installations. Available for
  MCR installations if provided by MCR.

``Matlab::mx``
  The mx library of Matlab (arrays), always available for MATLAB installations.
  Available for MCR installations if provided by MCR.

``Matlab::eng``
  Matlab engine library. Available only if the ``ENG_LIBRARY`` component
  is requested.

``Matlab::mat``
  Matlab matrix library. Available only if the ``MAT_LIBRARY`` component
  is requested.

``Matlab::MatlabEngine``
  Matlab C++ engine library, always available for MATLAB R2018a and newer.
  Available for MCR installations if provided by MCR.

``Matlab::MatlabDataArray``
  Matlab C++ data array library, always available for MATLAB R2018a and newer.
  Available for MCR installations if provided by MCR.

Variables defined by the module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Result variables
""""""""""""""""

``Matlab_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the (requested version of) Matlab installation
  was found.  All variables below are defined if Matlab is found.

``Matlab_VERSION``
  .. versionadded:: 3.27

  the numerical version (e.g. 23.2.0) of Matlab found. Not to be confused with
  Matlab release name (e.g. R2023b) that can be obtained with
  :command:`matlab_get_release_name_from_version`.
``Matlab_ROOT_DIR``
  the final root of the Matlab installation determined by the FindMatlab
  module.
``Matlab_MAIN_PROGRAM``
  the Matlab binary program. Available only if the component ``MAIN_PROGRAM``
  is given in the :command:`find_package` directive.
``Matlab_INCLUDE_DIRS``
 the path of the Matlab libraries headers
``Matlab_MEX_LIBRARY``
  library for mex, always available for MATLAB installations. Available for MCR
  installations if provided by MCR.
``Matlab_MX_LIBRARY``
  mx library of Matlab (arrays), always available for MATLAB installations.
  Available for MCR installations if provided by MCR.
``Matlab_ENG_LIBRARY``
  Matlab engine library. Available only if the component ``ENG_LIBRARY``
  is requested.
``Matlab_MAT_LIBRARY``
  Matlab matrix library. Available only if the component ``MAT_LIBRARY``
  is requested.
``Matlab_ENGINE_LIBRARY``
  .. versionadded:: 3.13

  Matlab C++ engine library, always available for MATLAB R2018a and newer.
  Available for MCR installations if provided by MCR.
``Matlab_DATAARRAY_LIBRARY``
  .. versionadded:: 3.13

  Matlab C++ data array library, always available for MATLAB R2018a and newer.
  Available for MCR installations if provided by MCR.
``Matlab_LIBRARIES``
  the whole set of libraries of Matlab
``Matlab_MEX_COMPILER``
  the mex compiler of Matlab. Currently not used.
  Available only if the component ``MEX_COMPILER`` is requested.
``Matlab_MCC_COMPILER``
  .. versionadded:: 3.13

  the mcc compiler of Matlab. Included with the Matlab Compiler add-on.
  Available only if the component ``MCC_COMPILER`` is requested.

Cached variables
""""""""""""""""

``Matlab_MEX_EXTENSION``
  the extension of the mex files for the current platform (given by Matlab).
``Matlab_ROOT_DIR``
  the location of the root of the Matlab installation found. If this value
  is changed by the user, the result variables are recomputed.

Provided commands
^^^^^^^^^^^^^^^^^

:command:`matlab_get_version_from_release_name`
  returns the version from the Matlab release name
:command:`matlab_get_release_name_from_version`
  returns the release name from the Matlab version
:command:`matlab_add_mex`
  adds a target compiling a MEX file.
:command:`matlab_add_unit_test`
  adds a Matlab unit test file as a test to the project.
:command:`matlab_extract_all_installed_versions_from_registry`
  parses the registry for all Matlab versions. Available on Windows only.
  The part of the registry parsed is dependent on the host processor
:command:`matlab_get_all_valid_matlab_roots_from_registry`
  returns all the possible Matlab or MCR paths, according to a previously
  given list. Only the existing/accessible paths are kept. This is mainly
  useful for the searching all possible Matlab installation.
:command:`matlab_get_mex_suffix`
  returns the suffix to be used for the mex files
  (platform/architecture dependent)
:command:`matlab_get_version_from_matlab_run`
  returns the version of Matlab/MCR, given the full directory of the Matlab/MCR
  installation path.


Known issues
^^^^^^^^^^^^

**Symbol clash in a MEX target**
  By default, every symbols inside a MEX
  file defined with the command :command:`matlab_add_mex` have hidden
  visibility, except for the entry point. This is the default behavior of
  the MEX compiler, which lowers the risk of symbol collision between the
  libraries shipped with Matlab, and the libraries to which the MEX file is
  linking to. This is also the default on Windows platforms.

  However, this is not sufficient in certain case, where for instance your
  MEX file is linking against libraries that are already loaded by Matlab,
  even if those libraries have different SONAMES.
  A possible solution is to hide the symbols of the libraries to which the
  MEX target is linking to. This can be achieved in GNU GCC compilers with
  the linker option ``-Wl,--exclude-libs,ALL``.

**Tests using GPU resources**
  in case your MEX file is using the GPU and
  in order to be able to run unit tests on this MEX file, the GPU resources
  should be properly released by Matlab. A possible solution is to make
  Matlab aware of the use of the GPU resources in the session, which can be
  performed by a command such as ``D = gpuDevice()`` at the beginning of
  the test script (or via a fixture).


Reference
^^^^^^^^^

.. variable:: Matlab_ROOT_DIR

   The root folder of the Matlab installation. If set before the call to
   :command:`find_package`, the module will look for the components in that
   path. If not set, then an automatic search of Matlab
   will be performed. If set, it should point to a valid version of Matlab.

.. variable:: MATLAB_FIND_DEBUG

   If set, the lookup of Matlab and the intermediate configuration steps are
   outputted to the console.

.. variable:: MATLAB_ADDITIONAL_VERSIONS

  If set, specifies additional versions of Matlab that may be looked for.
  The variable should be a list of strings, organized by pairs of release
  name and versions, such as follows:

  .. code-block:: cmake

    set(MATLAB_ADDITIONAL_VERSIONS
        "release_name1=corresponding_version1"
        "release_name2=corresponding_version2"
        ...
        )

  Example:

  .. code-block:: cmake

    set(MATLAB_ADDITIONAL_VERSIONS
        "R2013b=8.2"
        "R2013a=8.1"
        "R2012b=8.0")

  The order of entries in this list matters when several versions of
  Matlab are installed. The priority is set according to the ordering in
  this list.
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

set(_FindMatlab_SELF_DIR "${CMAKE_CURRENT_LIST_DIR}")

include(FindPackageHandleStandardArgs)

if(NOT WIN32 AND NOT APPLE AND NOT Threads_FOUND
    AND (CMAKE_C_COMPILER_LOADED OR CMAKE_CXX_COMPILER_LOADED))
  # MEX files use pthread if available
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads)
endif()

# The currently supported versions. Other version can be added by the user by
# providing MATLAB_ADDITIONAL_VERSIONS
if(NOT MATLAB_ADDITIONAL_VERSIONS)
  set(MATLAB_ADDITIONAL_VERSIONS)
endif()

set(MATLAB_VERSIONS_MAPPING
  "R2025a=25.1"
  "R2024b=24.2"
  "R2024a=24.1"
  "R2023b=23.2"
  "R2023a=9.14"
  "R2022b=9.13"
  "R2022a=9.12"
  "R2021b=9.11"
  "R2021a=9.10"
  "R2020b=9.9"
  "R2020a=9.8"
  "R2019b=9.7"
  "R2019a=9.6"
  "R2018b=9.5"
  "R2018a=9.4"
  "R2017b=9.3"
  "R2017a=9.2"
  "R2016b=9.1"
  "R2016a=9.0"
  "R2015b=8.6"
  "R2015a=8.5"
  "R2014b=8.4"
  "R2014a=8.3"
  "R2013b=8.2"
  "R2013a=8.1"
  "R2012b=8.0"
  "R2012a=7.14"
  "R2011b=7.13"
  "R2011a=7.12"
  "R2010b=7.11"

  ${MATLAB_ADDITIONAL_VERSIONS}
  )


# temporary folder for all Matlab runs
set(_matlab_temporary_folder ${CMAKE_BINARY_DIR}/Matlab)

file(MAKE_DIRECTORY "${_matlab_temporary_folder}")

#[=======================================================================[.rst:
.. command:: matlab_get_version_from_release_name

  .. code-block:: cmake

    matlab_get_version_from_release_name(release version)

  * Input: ``release`` is the release name (e.g. R2023b)
  * Output: ``version`` is the version of Matlab (e.g. 23.2.0)

  Returns the version of Matlab from a release name

  .. note::

    This command provides correct versions mappings for Matlab but not MCR.

#]=======================================================================]
macro(matlab_get_version_from_release_name release_name version_name)

  string(REGEX MATCHALL "${release_name}=([0-9]+\\.[0-9]+)" _matched ${MATLAB_VERSIONS_MAPPING})

  set(${version_name} "")
  if(NOT _matched STREQUAL "")
    set(${version_name} ${CMAKE_MATCH_1})
  else()
    message(WARNING "[MATLAB] The release name ${release_name} is not registered")
  endif()
  unset(_matched)

endmacro()


#[=======================================================================[.rst:
.. command:: matlab_get_release_name_from_version

  .. code-block:: cmake

    matlab_get_release_name_from_version(version release_name)

  * Input: ``version`` is the version of Matlab (e.g. 23.2.0)
  * Output: ``release_name`` is the release name (R2023b)

  Returns the release name from the version of Matlab

  .. note::

    This command provides correct version mappings for Matlab but not MCR.

#]=======================================================================]
function(matlab_get_release_name_from_version version release_name)

  # only the major.minor version is used
  string(REGEX REPLACE "^([0-9]+\\.[0-9]+).*" "\\1" version "${version}")

  foreach(_var IN LISTS MATLAB_VERSIONS_MAPPING)
    if(_var MATCHES "(.+)=${version}")
      set(${release_name} ${CMAKE_MATCH_1} PARENT_SCOPE)
      return()
    endif()
  endforeach()

  message(WARNING "[MATLAB] The version ${version} is not registered")

endfunction()


# extracts all the supported release names (R2022b...) of Matlab
# internal use
macro(matlab_get_supported_releases list_releases)
  set(${list_releases})
  foreach(_var IN LISTS MATLAB_VERSIONS_MAPPING)
    string(REGEX MATCHALL "(.+)=([0-9]+\\.[0-9]+)" _matched ${_var})
    if(NOT _matched STREQUAL "")
      list(APPEND ${list_releases} ${CMAKE_MATCH_1})
    endif()
    unset(_matched)
    unset(CMAKE_MATCH_1)
  endforeach()
  unset(_var)
endmacro()



# extracts all the supported versions of Matlab
# internal use
macro(matlab_get_supported_versions list_versions)
  set(${list_versions})
  foreach(_var IN LISTS MATLAB_VERSIONS_MAPPING)
    string(REGEX MATCHALL "(.+)=([0-9]+\\.[0-9]+)" _matched ${_var})
    if(NOT _matched STREQUAL "")
      list(APPEND ${list_versions} ${CMAKE_MATCH_2})
    endif()
    unset(_matched)
    unset(CMAKE_MATCH_1)
  endforeach()
  unset(_var)
endmacro()


#[=======================================================================[.rst:
.. command:: matlab_extract_all_installed_versions_from_registry

  This function parses the Windows registry and finds the Matlab versions that
  are installed. The found versions are stored in a given ``<versions-var>``.

  .. signature::
    matlab_extract_all_installed_versions_from_registry(<versions-var>
      [REGISTRY_VIEW view])
    :target: matlab_extract_all_installed_versions_from_registry-keyword

    .. versionadded:: 3.30

    * Output: ``<versions-var>`` is a list of all the versions of Matlab found
    * Input: ``REGISTRY_VIEW`` Optional registry view to use for registry
      interaction. The argument is passed (or omitted) to
      :command:`cmake_host_system_information` without further checks or
      modification.

  .. signature::
    matlab_extract_all_installed_versions_from_registry(<win64> <versions-var>)
    :target: matlab_extract_all_installed_versions_from_registry-positional

    * Input: ``win64`` is a boolean to search for the 64 bit version of
      Matlab. Set to ``ON`` to use the 64bit registry view or ``OFF`` to use the
      32bit registry view. If finer control is needed, see signature above.
    * Output: ``<versions-var>`` is a list of all the versions of Matlab found

  The returned list contains all versions under
  ``HKLM\SOFTWARE\Mathworks\MATLAB``,
  ``HKLM\SOFTWARE\Mathworks\MATLAB Runtime`` and
  ``HKLM\SOFTWARE\Mathworks\MATLAB Compiler Runtime`` or an empty list in
  case an error occurred (or nothing found).

  .. note::

    Only the versions are provided. No check is made over the existence of the
    installation referenced in the registry,

#]=======================================================================]
function(matlab_extract_all_installed_versions_from_registry win64_or_matlab_versions)

  if(NOT CMAKE_HOST_WIN32)
    message(FATAL_ERROR "[MATLAB] This function can only be called by a Windows host")
  endif()

  set(_registry_view_args)
  if("${ARGC}" EQUAL "2")
    # Old API: <win64> <matlab_versions>
    if(${win64_or_matlab_versions})
      set(_registry_view_args VIEW 64)
    else()
      set(_registry_view_args VIEW 32)
    endif()
    set(matlab_versions ${ARGV1})
  else()
    # New API: <matlab_versions> [REGISTRY_VIEW <view>]
    set(matlab_versions ${win64_or_matlab_versions})
    cmake_parse_arguments(_Matlab "" "REGISTRY_VIEW" "" ${ARGN})
    if(_Matlab_REGISTRY_VIEW)
      set(_registry_view_args VIEW "${_Matlab_REGISTRY_VIEW}")
    endif()
  endif()

  if(MATLAB_FIND_DEBUG)
    message(STATUS "[MATLAB] Extracting MATLAB versions with registry view args '${_registry_view_args}'")
  endif()

  set(matlabs_from_registry)

  foreach(_installation_type IN ITEMS "MATLAB" "MATLAB Runtime" "MATLAB Compiler Runtime")

    cmake_host_system_information(RESULT _reg
      QUERY WINDOWS_REGISTRY "HKLM/SOFTWARE/Mathworks/${_installation_type}"
      SUBKEYS
      ${_registry_view_args}
    )

    string(REGEX MATCHALL "([0-9]+(\\.[0-9]+)+)" _versions_regex "${_reg}")

    list(APPEND matlabs_from_registry ${_versions_regex})

  endforeach()

  if(matlabs_from_registry)
    list(REMOVE_DUPLICATES matlabs_from_registry)
    list(SORT matlabs_from_registry COMPARE NATURAL ORDER DESCENDING)
  endif()

  set(${matlab_versions} ${matlabs_from_registry} PARENT_SCOPE)

endfunction()



# (internal)
macro(extract_matlab_versions_from_registry_brute_force matlab_versions)
  # get the supported versions
  set(matlab_supported_versions)
  matlab_get_supported_versions(matlab_supported_versions)

  # we order from more recent to older
  if(matlab_supported_versions)
    list(REMOVE_DUPLICATES matlab_supported_versions)
    list(SORT matlab_supported_versions COMPARE NATURAL ORDER DESCENDING)
  endif()

  set(${matlab_versions} ${matlab_supported_versions})
endmacro()


#[=======================================================================[.rst:
.. command:: matlab_get_all_valid_matlab_roots_from_registry

  Populates the Matlab root with valid versions of Matlab or
  Matlab Runtime (MCR).
  The returned matlab_roots is organized in triplets
  ``(type,version_number,matlab_root_path)``, where ``type``
  indicates either ``MATLAB`` or ``MCR``.

  .. code-block:: cmake

    matlab_get_all_valid_matlab_roots_from_registry(matlab_versions matlab_roots [REGISTRY_VIEW view])

  * Input: ``matlab_versions`` of each of the Matlab or MCR installations
  * Output: ``matlab_roots`` location of each of the Matlab or MCR installations
  * Input: ``REGISTRY_VIEW`` Optional registry view to use for registry
    interaction. The argument is passed (or omitted) to
    :command:`cmake_host_system_information` without further checks or
    modification.

  .. versionadded:: 3.30
    The optional ``REGISTRY_VIEW`` argument was added to provide a more precise
    interface on how to interact with the Windows Registry.

#]=======================================================================]
function(matlab_get_all_valid_matlab_roots_from_registry matlab_versions matlab_roots)

  # The matlab_versions comes either from
  # extract_matlab_versions_from_registry_brute_force or
  # matlab_extract_all_installed_versions_from_registry.

  cmake_parse_arguments(_Matlab "" "REGISTRY_VIEW" "" ${ARGN})
  set(_registry_view_args)
  if(_Matlab_REGISTRY_VIEW)
    set(_registry_view_args VIEW "${_Matlab_REGISTRY_VIEW}")
  endif()
  if(MATLAB_FIND_DEBUG)
    message(STATUS "[MATLAB] Getting MATLAB roots with registry view args '${_registry_view_args}'")
  endif()

  # Mostly the major.minor version is used in Mathworks Windows Registry keys.
  # If the patch is not zero, major.minor.patch is used.
  list(TRANSFORM matlab_versions REPLACE "^([0-9]+\\.[0-9]+(\\.[1-9][0-9]*)?).*" "\\1")

  set(_matlab_roots_list )
  # check for Matlab installations
  foreach(_matlab_current_version IN LISTS matlab_versions)
    cmake_host_system_information(RESULT current_MATLAB_ROOT
      QUERY WINDOWS_REGISTRY "HKLM/SOFTWARE/Mathworks/MATLAB/${_matlab_current_version}"
      VALUE "MATLABROOT"
      ${_registry_view_args}
    )
    cmake_path(CONVERT "${current_MATLAB_ROOT}" TO_CMAKE_PATH_LIST current_MATLAB_ROOT)

    if(IS_DIRECTORY "${current_MATLAB_ROOT}")
      _Matlab_VersionInfoXML("${current_MATLAB_ROOT}" _matlab_version_tmp)
      if("${_matlab_version_tmp}" STREQUAL "unknown")
        set(_matlab_version_tmp ${_matlab_current_version})
      endif()
      list(APPEND _matlab_roots_list "MATLAB" ${_matlab_version_tmp} ${current_MATLAB_ROOT})
    endif()

  endforeach()

  # Check for MCR installations
  foreach(_installation_type IN ITEMS "MATLAB Runtime" "MATLAB Compiler Runtime")
    foreach(_matlab_current_version IN LISTS matlab_versions)
      cmake_host_system_information(RESULT current_MATLAB_ROOT
        QUERY WINDOWS_REGISTRY "HKLM/SOFTWARE/Mathworks/${_installation_type}/${_matlab_current_version}"
        VALUE "MATLABROOT"
        ${_registry_view_args}
      )
      cmake_path(CONVERT "${current_MATLAB_ROOT}" TO_CMAKE_PATH_LIST current_MATLAB_ROOT)

      # remove the dot
      string(REPLACE "." "" _matlab_current_version_without_dot "${_matlab_current_version}")

      if(IS_DIRECTORY "${current_MATLAB_ROOT}")
        if(IS_DIRECTORY "${current_MATLAB_ROOT}/v${_matlab_current_version_without_dot}")
          cmake_path(APPEND current_MATLAB_ROOT "v${_matlab_current_version_without_dot}")
        endif()
        _Matlab_VersionInfoXML("${current_MATLAB_ROOT}" _matlab_version_tmp)
        if("${_matlab_version_tmp}" STREQUAL "unknown")
          set(_matlab_version_tmp ${_matlab_current_version})
        endif()
        list(APPEND _matlab_roots_list "MCR" ${_matlab_version_tmp} "${current_MATLAB_ROOT}")
      endif()
    endforeach()
  endforeach()
  set(${matlab_roots} ${_matlab_roots_list} PARENT_SCOPE)
endfunction()

#[=======================================================================[.rst:
.. command:: matlab_get_mex_suffix

  Returns the extension of the mex files (the suffixes).
  This function should not be called before the appropriate Matlab root has
  been found.

  .. code-block:: cmake

    matlab_get_mex_suffix(matlab_root mex_suffix)

  * Input: ``matlab_root`` root of Matlab/MCR install e.g. ``Matlab_ROOT_DIR``
  * Output: ``mex_suffix`` variable name in which the suffix will be returned.
#]=======================================================================]
function(matlab_get_mex_suffix matlab_root mex_suffix)

  # find_program does not consider script suffix .bat for Matlab mexext.bat on Windows
  set(mexext_suffix "")
  if(WIN32)
    set(mexext_suffix ".bat")
  endif()

  find_program(
    Matlab_MEXEXTENSIONS_PROG
    NAMES mexext mexext${mexext_suffix}
    PATHS ${matlab_root}/bin
    DOC "Matlab MEX extension provider"
    NO_DEFAULT_PATH
  )

  if(MATLAB_FIND_DEBUG)
    message(STATUS "[MATLAB] Determining mex files extensions from '${matlab_root}/bin' with program '${Matlab_MEXEXTENSIONS_PROG}'")
  endif()

  # the program has been found?
  if(NOT Matlab_MEXEXTENSIONS_PROG)
    if(MATLAB_FIND_DEBUG)
      message(WARNING "[MATLAB] Cannot found mexext program. Matlab root is ${matlab_root}")
    endif()
    unset(Matlab_MEXEXTENSIONS_PROG CACHE)
    return()
  endif()

  set(_matlab_mex_extension)

  set(devnull)
  if(UNIX)
    set(devnull INPUT_FILE /dev/null)
  elseif(WIN32)
    set(devnull INPUT_FILE NUL)
  endif()

  set(_arch)
  if(WIN32)
    # this environment variable is used to determine the arch on Windows
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(_arch "MATLAB_ARCH=win64")
    else()
      set(_arch "MATLAB_ARCH=win32")
    endif()
  endif()

  # this is the preferred way. If this does not work properly (eg. MCR on Windows), then we use our own knowledge
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env ${_arch} ${Matlab_MEXEXTENSIONS_PROG}
    OUTPUT_VARIABLE _matlab_mex_extension
    ERROR_VARIABLE _matlab_mex_extension_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ${devnull})

  if(_matlab_mex_extension_error)
    if(WIN32)
      # this is only for intel architecture
      if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_matlab_mex_extension "mexw64")
      else()
        set(_matlab_mex_extension "mexw32")
      endif()
    endif()
  endif()

  string(STRIP "${_matlab_mex_extension}"  _matlab_mex_extension)
  if(MATLAB_FIND_DEBUG)
    message(STATUS "[MATLAB] '${Matlab_MEXEXTENSIONS_PROG}' : determined extension '${_matlab_mex_extension}' and error string is '${_matlab_mex_extension_error}'")
  endif()

  set(${mex_suffix} ${_matlab_mex_extension} PARENT_SCOPE)
endfunction()


#[=======================================================================[.rst:
.. command:: matlab_get_version_from_matlab_run

  This function runs Matlab program specified on arguments and extracts its
  version. If the path provided for the Matlab installation points to an MCR
  installation, the version is extracted from the installed files.

  .. code-block:: cmake

    matlab_get_version_from_matlab_run(matlab_binary_path matlab_list_versions)

  * Input: ``matlab_binary_path`` path of the `matlab` binary executable
  * Output: ``matlab_list_versions`` the version extracted from Matlab
#]=======================================================================]
function(matlab_get_version_from_matlab_run matlab_binary_program matlab_list_versions)

  set(${matlab_list_versions} "" PARENT_SCOPE)

  if(MATLAB_FIND_DEBUG)
    message(STATUS "[MATLAB] Determining the version of Matlab from ${matlab_binary_program}")
  endif()

  if(EXISTS "${_matlab_temporary_folder}/matlabVersionLog.cmaketmp")
    if(MATLAB_FIND_DEBUG)
      message(STATUS "[MATLAB] Removing previous ${_matlab_temporary_folder}/matlabVersionLog.cmaketmp file")
    endif()
    file(REMOVE "${_matlab_temporary_folder}/matlabVersionLog.cmaketmp")
  endif()


  # the log file is needed since on windows the command executes in a new
  # window and it is not possible to get back the answer of Matlab
  # the -wait command is needed on windows, otherwise the call returns
  # immediately after the program launches itself.
  if(WIN32)
    set(_matlab_additional_commands "-wait")
  endif()

  set(devnull)
  if(UNIX)
    set(devnull INPUT_FILE /dev/null)
  elseif(WIN32)
    set(devnull INPUT_FILE NUL)
  endif()

  # we first try to run a simple program using the -r option, and then we use the
  # -batch option that is supported and recommended since R2019a
  set(_matlab_get_version_failed_with_r_option FALSE)

  # timeout set to 120 seconds, in case it does not start
  # note as said before OUTPUT_VARIABLE cannot be used in a platform
  # independent manner however, not setting it would flush the output of Matlab
  # in the current console (unix variant)
  execute_process(
    COMMAND "${matlab_binary_program}" -nosplash -nojvm ${_matlab_additional_commands} -logfile "matlabVersionLog.cmaketmp" -nodesktop -nodisplay -r "version, exit"
    OUTPUT_VARIABLE _matlab_version_from_cmd_dummy
    RESULT_VARIABLE _matlab_result_version_call
    ERROR_VARIABLE _matlab_result_version_call_error
    TIMEOUT 120
    WORKING_DIRECTORY "${_matlab_temporary_folder}"
    ${devnull}
    )

  if(_matlab_result_version_call MATCHES "timeout")
    if(MATLAB_FIND_DEBUG)
      message(WARNING "[MATLAB] Unable to determine the version of Matlab."
        " Matlab call with -r option timed out after 120 seconds.")
    endif()
    set(_matlab_get_version_failed_with_r_option TRUE)
  endif()

  if(NOT ${_matlab_get_version_failed_with_r_option} AND ${_matlab_result_version_call})
    if(MATLAB_FIND_DEBUG)
      message(WARNING "[MATLAB] Unable to determine the version of Matlab. Matlab call with -r option returned with error ${_matlab_result_version_call}.")
    endif()
    set(_matlab_get_version_failed_with_r_option TRUE)
  elseif(NOT ${_matlab_get_version_failed_with_r_option} AND NOT EXISTS "${_matlab_temporary_folder}/matlabVersionLog.cmaketmp")
    if(MATLAB_FIND_DEBUG)
      message(WARNING "[MATLAB] Unable to determine the version of Matlab. The log file does not exist.")
    endif()
    set(_matlab_get_version_failed_with_r_option TRUE)
  endif()

  if(_matlab_get_version_failed_with_r_option)
    execute_process(
      COMMAND "${matlab_binary_program}" -nosplash -nojvm ${_matlab_additional_commands} -logfile "matlabVersionLog.cmaketmp" -nodesktop -nodisplay -batch "version, exit"
      OUTPUT_VARIABLE _matlab_version_from_cmd_dummy_batch
      RESULT_VARIABLE _matlab_result_version_call_batch
      ERROR_VARIABLE _matlab_result_version_call_error_batch
      TIMEOUT 120
      WORKING_DIRECTORY "${_matlab_temporary_folder}"
      ${devnull}
      )

    if(_matlab_result_version_call_batch MATCHES "timeout")
      if(MATLAB_FIND_DEBUG)
        message(WARNING "[MATLAB] Unable to determine the version of Matlab."
          " Matlab call with -batch option timed out after 120 seconds.")
      endif()
      return()
    endif()

    if(${_matlab_result_version_call_batch})
      if(MATLAB_FIND_DEBUG)
        message(WARNING "[MATLAB] Command executed \"${matlab_binary_program}\" -nosplash -nojvm ${_matlab_additional_commands} -logfile \"matlabVersionLog.cmaketmp\" -nodesktop -nodisplay -batch \"version, exit\"")
        message(WARNING "_matlab_version_from_cmd_dummy_batch (OUTPUT_VARIABLE): ${_matlab_version_from_cmd_dummy_batch}")
        message(WARNING "_matlab_result_version_call_batch (RESULT_VARIABLE): ${_matlab_result_version_call_batch}")
        message(WARNING "_matlab_result_version_call_error_batch (ERROR_VARIABLE): ${_matlab_result_version_call_error_batch}")
        message(WARNING "[MATLAB] Unable to determine the version of Matlab. Matlab call with -batch option returned with error ${_matlab_result_version_call_batch}.")
      endif()
      return()
    elseif(NOT ${_matlab_get_version_failed_with_r_option} AND NOT EXISTS "${_matlab_temporary_folder}/matlabVersionLog.cmaketmp")
      if(MATLAB_FIND_DEBUG)
        message(WARNING "[MATLAB] Unable to determine the version of Matlab. The log file does not exist.")
      endif()
      return()
    endif()
  endif()

  if(NOT EXISTS "${_matlab_temporary_folder}/matlabVersionLog.cmaketmp")
    # last resort check as some HPC with "module load matlab" not enacted fail to catch in earlier checks
    # and error CMake configure even if find_package(Matlab) is not REQUIRED
    if(MATLAB_FIND_DEBUG)
      message(WARNING "[MATLAB] Unable to determine the version of Matlab. The version log file does not exist.")
    endif()
    return()
  endif()

  # if successful, read back the log
  file(READ "${_matlab_temporary_folder}/matlabVersionLog.cmaketmp" _matlab_version_from_cmd)
  file(REMOVE "${_matlab_temporary_folder}/matlabVersionLog.cmaketmp")

  set(index -1)
  string(FIND "${_matlab_version_from_cmd}" "ans" index)
  if(index EQUAL -1)

    if(MATLAB_FIND_DEBUG)
      message(WARNING "[MATLAB] Cannot find the version of Matlab returned by the run.")
    endif()

  else()
    set(matlab_list_of_all_versions_tmp)

    string(SUBSTRING "${_matlab_version_from_cmd}" ${index} -1 substring_ans)
    string(
      REGEX MATCHALL "ans[\r\n\t ]*=[\r\n\t ]*'?([0-9]+(\\.[0-9]+)+)"
      matlab_versions_regex
      ${substring_ans})
    foreach(match IN LISTS matlab_versions_regex)
      string(
        REGEX MATCH "ans[\r\n\t ]*=[\r\n\t ]*'?(([0-9]+)(\\.([0-9]+))?)"
        current_match ${match})

      list(APPEND matlab_list_of_all_versions_tmp ${CMAKE_MATCH_1})
    endforeach()
    if(matlab_list_of_all_versions_tmp)
      list(REMOVE_DUPLICATES matlab_list_of_all_versions_tmp)
    endif()
    set(${matlab_list_versions} ${matlab_list_of_all_versions_tmp} PARENT_SCOPE)

  endif()

endfunction()

#[=======================================================================[.rst:
.. command:: matlab_add_unit_test

  Adds a Matlab unit test to the test set of cmake/ctest.
  This command requires the component ``MAIN_PROGRAM`` and hence is not
  available for an MCR installation.

  The unit test uses the Matlab unittest framework (default, available
  starting Matlab 2013b+) except if the option ``NO_UNITTEST_FRAMEWORK``
  is given.

  The function expects one Matlab test script file to be given.
  In the case ``NO_UNITTEST_FRAMEWORK`` is given, the unittest script file
  should contain the script to be run, plus an exit command with the exit
  value. This exit value will be passed to the ctest framework (0 success,
  non 0 failure). Additional arguments accepted by :command:`add_test` can be
  passed through ``TEST_ARGS`` (eg. ``CONFIGURATION <config> ...``).

  .. code-block:: cmake

    matlab_add_unit_test(
        NAME <name>
        UNITTEST_FILE matlab_file_containing_unittest.m
        [CUSTOM_TEST_COMMAND matlab_command_to_run_as_test]
        [UNITTEST_PRECOMMAND matlab_command_to_run]
        [TIMEOUT timeout]
        [ADDITIONAL_PATH path1 [path2 ...]]
        [MATLAB_ADDITIONAL_STARTUP_OPTIONS option1 [option2 ...]]
        [TEST_ARGS arg1 [arg2 ...]]
        [NO_UNITTEST_FRAMEWORK]
        )

  Function Parameters:

  ``NAME``
    name of the unittest in ctest.
  ``UNITTEST_FILE``
    the matlab unittest file. Its path will be automatically
    added to the Matlab path.
  ``CUSTOM_TEST_COMMAND``
    Matlab script command to run as the test.
    If this is not set, then the following is run:
    ``runtests('matlab_file_name'), exit(max([ans(1,:).Failed]))``
    where ``matlab_file_name`` is the ``UNITTEST_FILE`` without the extension.
  ``UNITTEST_PRECOMMAND``
    Matlab script command to be ran before the file
    containing the test (eg. GPU device initialization based on CMake
    variables).
  ``TIMEOUT``
    the test timeout in seconds. Defaults to 180 seconds as the
    Matlab unit test may hang.
  ``ADDITIONAL_PATH``
    a list of paths to add to the Matlab path prior to
    running the unit test.
  ``MATLAB_ADDITIONAL_STARTUP_OPTIONS``
    a list of additional option in order
    to run Matlab from the command line.
    ``-nosplash -nodesktop -nodisplay`` are always added.
  ``TEST_ARGS``
    Additional options provided to the add_test command. These
    options are added to the default options (eg. "CONFIGURATIONS Release")
  ``NO_UNITTEST_FRAMEWORK``
    when set, indicates that the test should not
    use the unittest framework of Matlab (available for versions >= R2013a).
  ``WORKING_DIRECTORY``
    This will be the working directory for the test. If specified it will
    also be the output directory used for the log file of the test run.
    If not specified the temporary directory ``${CMAKE_BINARY_DIR}/Matlab`` will
    be used as the working directory and the log location.

#]=======================================================================]
function(matlab_add_unit_test)

  if(NOT Matlab_MAIN_PROGRAM)
    message(FATAL_ERROR "[MATLAB] This functionality needs the MAIN_PROGRAM component (not default)")
  endif()

  set(options NO_UNITTEST_FRAMEWORK)
  set(oneValueArgs NAME UNITTEST_FILE TIMEOUT WORKING_DIRECTORY
    UNITTEST_PRECOMMAND CUSTOM_TEST_COMMAND)
  set(multiValueArgs ADDITIONAL_PATH MATLAB_ADDITIONAL_STARTUP_OPTIONS TEST_ARGS)

  set(prefix _matlab_unittest_prefix)
  cmake_parse_arguments(PARSE_ARGV 0 ${prefix} "${options}" "${oneValueArgs}" "${multiValueArgs}" )

  if(NOT ${prefix}_NAME)
    message(FATAL_ERROR "[MATLAB] The Matlab test name cannot be empty")
  endif()

  # The option to run a batch program with MATLAB changes depending on the MATLAB version
  # For MATLAB before R2019a (9.6), the only supported option is -r, afterwards the suggested option
  # is -batch as -r is deprecated
  set(maut_BATCH_OPTION "-r")
  if(NOT (Matlab_VERSION_STRING STREQUAL ""))
    if(Matlab_VERSION_STRING VERSION_GREATER_EQUAL "9.6")
      set(maut_BATCH_OPTION "-batch")
    endif()
  endif()

  # The ${${prefix}_TEST_ARGS} and ${${prefix}_UNPARSED_ARGUMENTS} used below
  # should have semicolons escaped, so empty arguments should be preserved.
  # There's also no target used for the command, so we don't need to do
  # anything here for CMP0178.
  add_test(NAME ${${prefix}_NAME}
           COMMAND ${CMAKE_COMMAND}
            "-Dtest_name=${${prefix}_NAME}"
            "-Dadditional_paths=${${prefix}_ADDITIONAL_PATH}"
            "-Dtest_timeout=${${prefix}_TIMEOUT}"
            "-Doutput_directory=${_matlab_temporary_folder}"
            "-Dworking_directory=${${prefix}_WORKING_DIRECTORY}"
            "-DMatlab_PROGRAM=${Matlab_MAIN_PROGRAM}"
            "-Dno_unittest_framework=${${prefix}_NO_UNITTEST_FRAMEWORK}"
            "-DMatlab_ADDITIONAL_STARTUP_OPTIONS=${${prefix}_MATLAB_ADDITIONAL_STARTUP_OPTIONS}"
            "-Dunittest_file_to_run=${${prefix}_UNITTEST_FILE}"
            "-Dcustom_Matlab_test_command=${${prefix}_CUSTOM_TEST_COMMAND}"
            "-Dcmd_to_run_before_test=${${prefix}_UNITTEST_PRECOMMAND}"
            "-Dmaut_BATCH_OPTION=${maut_BATCH_OPTION}"
            -P ${_FindMatlab_SELF_DIR}/MatlabTestsRedirect.cmake
           ${${prefix}_TEST_ARGS}
           ${${prefix}_UNPARSED_ARGUMENTS}
           )
endfunction()


#[=======================================================================[.rst:
.. command:: matlab_add_mex

  Adds a Matlab MEX target.
  This commands compiles the given sources with the current tool-chain in
  order to produce a MEX file. The final name of the produced output may be
  specified, as well as additional link libraries, and a documentation entry
  for the MEX file. Remaining arguments of the call are passed to the
  :command:`add_library` or :command:`add_executable` command.

  .. code-block:: cmake

     matlab_add_mex(
         NAME <name>
         [EXECUTABLE | MODULE | SHARED]
         SRC src1 [src2 ...]
         [OUTPUT_NAME output_name]
         [DOCUMENTATION file.txt]
         [LINK_TO target1 target2 ...]
         [R2017b | R2018a]
         [EXCLUDE_FROM_ALL]
         [NO_IMPLICIT_LINK_TO_MATLAB_LIBRARIES]
         [...]
     )

  Function Parameters:

  ``NAME``
    name of the target.
  ``SRC``
    list of source files.
  ``LINK_TO``
    a list of additional link dependencies.  The target links to ``libmex``
    and ``libmx`` by default, unless the
    ``NO_IMPLICIT_LINK_TO_MATLAB_LIBRARIES`` option is passed.
  ``OUTPUT_NAME``
    if given, overrides the default name. The default name is
    the name of the target without any prefix and
    with ``Matlab_MEX_EXTENSION`` suffix.
  ``DOCUMENTATION``
    if given, the file ``file.txt`` will be considered as
    being the documentation file for the MEX file. This file is copied into
    the same folder without any processing, with the same name as the final
    mex file, and with extension `.m`. In that case, typing ``help <name>``
    in Matlab prints the documentation contained in this file.
  ``R2017b`` or ``R2018a``
    .. versionadded:: 3.14

    May be given to specify the version of the C API
    to use: ``R2017b`` specifies the traditional (separate complex) C API,
    and corresponds to the ``-R2017b`` flag for the `mex` command. ``R2018a``
    specifies the new interleaved complex C API, and corresponds to the
    ``-R2018a`` flag for the `mex` command. Ignored if MATLAB version prior
    to R2018a. Defaults to ``R2017b``.

  ``MODULE`` or ``SHARED``
    .. versionadded:: 3.7

    May be given to specify the type of library to be
    created.

  ``EXECUTABLE``
    .. versionadded:: 3.7

    May be given to create an executable instead of
    a library. If no type is given explicitly, the type is ``SHARED``.
  ``EXCLUDE_FROM_ALL``
    This option has the same meaning as for :prop_tgt:`EXCLUDE_FROM_ALL` and
    is forwarded to :command:`add_library` or :command:`add_executable`
    commands.
  ``NO_IMPLICIT_LINK_TO_MATLAB_LIBRARIES``
    .. versionadded:: 3.24

    This option permits to disable the automatic linking of MATLAB
    libraries, so that only the libraries that are actually required can be
    linked via the ``LINK_TO`` option.

  The documentation file is not processed and should be in the following
  format:

  ::

    % This is the documentation
    function ret = mex_target_output_name(input1)

#]=======================================================================]
function(matlab_add_mex)

  set(options EXECUTABLE MODULE SHARED R2017b R2018a EXCLUDE_FROM_ALL NO_IMPLICIT_LINK_TO_MATLAB_LIBRARIES)
  set(oneValueArgs NAME DOCUMENTATION OUTPUT_NAME)
  set(multiValueArgs LINK_TO SRC)

  set(prefix _matlab_addmex_prefix)
  cmake_parse_arguments(${prefix} "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  if(NOT ${prefix}_NAME)
    message(FATAL_ERROR "[MATLAB] The MEX target name cannot be empty")
  endif()

  if(NOT ${prefix}_OUTPUT_NAME)
    set(${prefix}_OUTPUT_NAME ${${prefix}_NAME})
  endif()

  if(NOT Matlab_VERSION_STRING VERSION_LESS "9.1") # For 9.1 (R2016b) and newer, add version source file
    # Compilers officially supported by Matlab 9.1 (R2016b):
    #   MinGW 4.9, MSVC 2012, Intel C++ 2013, Xcode 6, GCC 4.9
    # These compilers definitely support the -w flag to suppress warnings.
    # Other compilers (Clang) may support the -w flag and can be added here.
    set(_Matlab_silenceable_compilers AppleClang Clang GNU Intel IntelLLVM MSVC)

    # Add the correct version file depending on which languages are enabled in the project
    if(CMAKE_C_COMPILER_LOADED)
      # If C is enabled, use the .c file as it will work fine also with C++
      set(MEX_VERSION_FILE "${Matlab_ROOT_DIR}/extern/version/c_mexapi_version.c")
      # Silence warnings for version source file
      if("${CMAKE_C_COMPILER_ID}" IN_LIST _Matlab_silenceable_compilers)
        set_source_files_properties("${MEX_VERSION_FILE}" PROPERTIES COMPILE_OPTIONS -w)
      endif()
    elseif(CMAKE_CXX_COMPILER_LOADED)
      # If C is not enabled, check if CXX is enabled and use the .cpp file
      # to avoid that the .c file is silently ignored
      set(MEX_VERSION_FILE "${Matlab_ROOT_DIR}/extern/version/cpp_mexapi_version.cpp")
      if("${CMAKE_CXX_COMPILER_ID}" IN_LIST _Matlab_silenceable_compilers)
        set_source_files_properties("${MEX_VERSION_FILE}" PROPERTIES COMPILE_OPTIONS -w)
      endif()
    else()
      # If neither C or CXX is enabled, warn because we cannot add the source.
      # TODO: add support for fortran mex files
      message(WARNING "[MATLAB] matlab_add_mex requires that at least C or CXX are enabled languages")
    endif()
  endif()

  # For 9.4 (R2018a) and newer, add API macro.
  # Add it for unknown versions too, just in case.
  if(NOT Matlab_VERSION_STRING VERSION_LESS "9.4"
      OR Matlab_VERSION_STRING STREQUAL "unknown")
    if(${${prefix}_R2018a})
      set(MEX_API_MACRO "MATLAB_DEFAULT_RELEASE=R2018a")
    else()
      set(MEX_API_MACRO "MATLAB_DEFAULT_RELEASE=R2017b")
    endif()
  endif()

  set(_option_EXCLUDE_FROM_ALL)
  if(${prefix}_EXCLUDE_FROM_ALL)
    set(_option_EXCLUDE_FROM_ALL "EXCLUDE_FROM_ALL")
  endif()

  if(${prefix}_EXECUTABLE)
    add_executable(${${prefix}_NAME}
      ${_option_EXCLUDE_FROM_ALL}
      ${${prefix}_SRC}
      ${MEX_VERSION_FILE}
      ${${prefix}_DOCUMENTATION}
      ${${prefix}_UNPARSED_ARGUMENTS})
  else()
    if(${prefix}_MODULE)
      set(type MODULE)
    else()
      set(type SHARED)
    endif()

    add_library(${${prefix}_NAME}
      ${type}
      ${_option_EXCLUDE_FROM_ALL}
      ${${prefix}_SRC}
      ${MEX_VERSION_FILE}
      ${${prefix}_DOCUMENTATION}
      ${${prefix}_UNPARSED_ARGUMENTS})
  endif()

  target_include_directories(${${prefix}_NAME} SYSTEM PRIVATE ${Matlab_INCLUDE_DIRS})

  if(NOT ${prefix}_NO_IMPLICIT_LINK_TO_MATLAB_LIBRARIES)
    if(Matlab_HAS_CPP_API)
      if(Matlab_ENGINE_LIBRARY)
        target_link_libraries(${${prefix}_NAME} ${Matlab_ENGINE_LIBRARY})
      endif()
      if(Matlab_DATAARRAY_LIBRARY)
        target_link_libraries(${${prefix}_NAME} ${Matlab_DATAARRAY_LIBRARY})
      endif()
    endif()

    target_link_libraries(${${prefix}_NAME} ${Matlab_MEX_LIBRARY} ${Matlab_MX_LIBRARY})
  endif()
  target_link_libraries(${${prefix}_NAME} ${${prefix}_LINK_TO})
  set_target_properties(${${prefix}_NAME}
      PROPERTIES
        PREFIX ""
        OUTPUT_NAME ${${prefix}_OUTPUT_NAME}
        SUFFIX ".${Matlab_MEX_EXTENSION}")

  target_compile_definitions(${${prefix}_NAME} PRIVATE ${MEX_API_MACRO} MATLAB_MEX_FILE)

  # documentation
  if(NOT ${${prefix}_DOCUMENTATION} STREQUAL "")
    get_target_property(output_name ${${prefix}_NAME} OUTPUT_NAME)
    add_custom_command(
      TARGET ${${prefix}_NAME}
      PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${${prefix}_DOCUMENTATION} $<TARGET_FILE_DIR:${${prefix}_NAME}>/${output_name}.m
      COMMENT "[MATLAB] Copy ${${prefix}_NAME} documentation file into the output folder"
    )
  endif() # documentation

  # entry point in the mex file + taking care of visibility and symbol clashes.
  if(WIN32)

    if (MSVC)

      string(APPEND _link_flags " /EXPORT:mexFunction")
      if(NOT Matlab_VERSION_STRING VERSION_LESS "9.1") # For 9.1 (R2016b) and newer, export version
        string(APPEND _link_flags " /EXPORT:mexfilerequiredapiversion")
      endif()

      set_property(TARGET ${${prefix}_NAME} APPEND PROPERTY LINK_FLAGS ${_link_flags})

    endif() # No other compiler currently supported on Windows.

    set_property(TARGET ${${prefix}_NAME} PROPERTY
      DEFINE_SYMBOL "DLL_EXPORT_SYM=__declspec(dllexport)")

  else()

    if(Matlab_VERSION_STRING VERSION_LESS "9.1") # For versions prior to 9.1 (R2016b)
      set(_ver_map_files ${Matlab_EXTERN_LIBRARY_DIR}/mexFunction.map)
    else()                                          # For 9.1 (R2016b) and newer
      set(_ver_map_files ${Matlab_EXTERN_LIBRARY_DIR}/c_exportsmexfileversion.map)
    endif()

    if(NOT Matlab_VERSION_STRING VERSION_LESS "9.5") # For 9.5 (R2018b) (and newer?)
      target_compile_options(${${prefix}_NAME} PRIVATE "-fvisibility=default")
      # This one is weird, it might be a bug in <mex.h> for R2018b. When compiling with
      # -fvisibility=hidden, the symbol `mexFunction` cannot be exported. Reading the
      # source code for <mex.h>, it seems that the preprocessor macro `MW_NEEDS_VERSION_H`
      # needs to be defined for `__attribute__((visibility("default")))` to be added
      # in front of the declaration of `mexFunction`. In previous versions of MATLAB this
      # was not the case, there `DLL_EXPORT_SYM` needed to be defined.
      # Adding `-fvisibility=hidden` to the `mex` command causes the build to fail.
      # TODO: Check that this is still necessary in R2019a when it comes out.
    endif()

    if(APPLE)

      if(Matlab_HAS_CPP_API)
        list(APPEND _ver_map_files ${Matlab_EXTERN_LIBRARY_DIR}/cppMexFunction.map) # This one doesn't exist on Linux
        string(APPEND _link_flags " -Wl,-U,_mexCreateMexFunction -Wl,-U,_mexDestroyMexFunction -Wl,-U,_mexFunctionAdapter")
        # On MacOS, the MEX command adds the above, without it the link breaks
        # because we indiscriminately use "cppMexFunction.map" even for C API MEX-files.
      endif()

      set(_export_flag_name -exported_symbols_list)

    else() # Linux

      if(Threads_FOUND)
        target_link_libraries(${${prefix}_NAME} Threads::Threads)
      endif()

      string(APPEND _link_flags " -Wl,--as-needed")

      set(_export_flag_name --version-script)

    endif()

    foreach(_file IN LISTS _ver_map_files)
      string(APPEND _link_flags " -Wl,${_export_flag_name},${_file}")
    endforeach()

    # The `mex` command doesn't add this define. It is specified here in order
    # to export the symbol in case the client code decides to hide its symbols
    set_target_properties(${${prefix}_NAME}
      PROPERTIES
        DEFINE_SYMBOL "DLL_EXPORT_SYM=__attribute__((visibility(\"default\")))"
        LINK_FLAGS "${_link_flags}"
    )

  endif()

endfunction()


# (internal)
# Used to get the version of matlab, using caching. This basically transforms the
# output of the root list, with possible unknown version, to a version
# This can possibly run Matlab for extracting the version.
function(_Matlab_get_version_from_root matlab_root matlab_or_mcr matlab_known_version matlab_final_version)

  # if the version is not trivial, we query matlab (if not MCR) for that
  # we keep track of the location of matlab that induced this version
  #if(NOT DEFINED Matlab_PROG_VERSION_STRING_AUTO_DETECT)
  #  set(Matlab_PROG_VERSION_STRING_AUTO_DETECT "" CACHE INTERNAL "internal matlab location for the discovered version")
  #endif()

  if(NOT matlab_or_mcr STREQUAL "UNKNOWN")
    set(Matlab_OR_MCR_INTERNAL ${matlab_or_mcr} CACHE INTERNAL "Whether Matlab root contains MATLAB or MCR")
  endif()

  if(NOT matlab_known_version STREQUAL "NOTFOUND")
    # the version is known, we just return it
    set(${matlab_final_version} ${matlab_known_version} PARENT_SCOPE)
    set(Matlab_VERSION_STRING_INTERNAL ${matlab_known_version} CACHE INTERNAL "Matlab version (automatically determined)")
    return()
  endif()

  if(matlab_or_mcr STREQUAL "UNKNOWN")
    if(MATLAB_FIND_DEBUG)
      message(WARNING "[MATLAB] Determining Matlab or MCR")
    endif()

    if(EXISTS "${matlab_root}/appdata/version.xml")
      # we inspect the application version.xml file that contains the product information
      file(READ "${matlab_root}/appdata/version.xml" productinfo_string)
      string(REGEX MATCH "<installedProductData.*displayedString=\"([a-zA-Z ]+)\".*/>"
             product_reg_match
             ${productinfo_string}
            )

      # default fallback to Matlab
      set(matlab_or_mcr "MATLAB")
      if(NOT CMAKE_MATCH_1 STREQUAL "")
        string(TOLOWER "${CMAKE_MATCH_1}" product_reg_match)

        if(product_reg_match STREQUAL "matlab runtime")
          set(matlab_or_mcr "MCR")
        endif()
      endif()
    endif()

    if(MATLAB_FIND_DEBUG)
      message(WARNING "[MATLAB] '${matlab_root}' contains the '${matlab_or_mcr}'")
    endif()
  endif()

  # UNKNOWN is the default behavior in case we
  # - have an erroneous matlab_root
  # - have an initial 'UNKNOWN'
  if(matlab_or_mcr STREQUAL "MATLAB" OR matlab_or_mcr STREQUAL "UNKNOWN")
    # MATLAB versions
    set(_matlab_current_program ${Matlab_MAIN_PROGRAM})

    # do we already have a matlab program?
    if(NOT _matlab_current_program)

      set(_find_matlab_options)
      if(IS_DIRECTORY "${matlab_root}")
        set(_find_matlab_options PATHS ${matlab_root} ${matlab_root}/bin NO_DEFAULT_PATH)
      endif()

      find_program(
          _matlab_current_program
          NAMES matlab
          ${_find_matlab_options}
          DOC "Matlab main program"
        )
    endif()

    if(NOT _matlab_current_program)
      # if not found, clear the dependent variables
      if(MATLAB_FIND_DEBUG)
        message(WARNING "[MATLAB] Cannot find the main matlab program under ${matlab_root}")
      endif()
      set(Matlab_PROG_VERSION_STRING_AUTO_DETECT "" CACHE INTERNAL "internal matlab location for the discovered version")
      set(Matlab_VERSION_STRING_INTERNAL "" CACHE INTERNAL "internal matlab location for the discovered version")
      set(Matlab_OR_MCR_INTERNAL ${matlab_or_mcr} CACHE INTERNAL "Whether Matlab root contains MATLAB or MCR")
      unset(_matlab_current_program)
      unset(_matlab_current_program CACHE)
      return()
    endif()

    # full real path for path comparison
    get_filename_component(_matlab_main_real_path_tmp "${_matlab_current_program}" REALPATH)
    unset(_matlab_current_program)
    unset(_matlab_current_program CACHE)

    # is it the same as the previous one?
    if(_matlab_main_real_path_tmp STREQUAL Matlab_PROG_VERSION_STRING_AUTO_DETECT)
      set(${matlab_final_version} ${Matlab_VERSION_STRING_INTERNAL} PARENT_SCOPE)
      return()
    endif()

    # update the location of the program
    set(Matlab_PROG_VERSION_STRING_AUTO_DETECT
        ${_matlab_main_real_path_tmp}
        CACHE INTERNAL "internal matlab location for the discovered version")

    _Matlab_VersionInfoXML("${matlab_root}" _matlab_version_tmp)
    if(NOT "${_matlab_version_tmp}" STREQUAL "unknown")
      # at least back to R2016 VersionInfo.xml exists
      set(matlab_list_of_all_versions ${_matlab_version_tmp})
    else()
      # time consuming, less stable way to find Matlab version by running Matlab
      matlab_get_version_from_matlab_run("${Matlab_PROG_VERSION_STRING_AUTO_DETECT}" matlab_list_of_all_versions)
    endif()

    list(LENGTH matlab_list_of_all_versions list_of_all_versions_length)
    if(list_of_all_versions_length GREATER 0)
      list(GET matlab_list_of_all_versions 0 _matlab_version_tmp)
    else()
      set(_matlab_version_tmp "unknown")
    endif()

    # set the version into the cache
    set(Matlab_VERSION_STRING_INTERNAL ${_matlab_version_tmp} CACHE INTERNAL "Matlab version (automatically determined)")
    set(Matlab_OR_MCR_INTERNAL ${matlab_or_mcr} CACHE INTERNAL "Whether Matlab root contains MATLAB or MCR")

    # warning, just in case several versions found (should not happen)
    if((list_of_all_versions_length GREATER 1) AND MATLAB_FIND_DEBUG)
      message(WARNING "[MATLAB] Found several versions, taking the first one (versions found ${matlab_list_of_all_versions})")
    endif()

  else()
    # MCR
    # we cannot run anything in order to extract the version. We assume that the file
    # VersionInfo.xml exists under the MatlabRoot, we look for it and extract the version from there
    _Matlab_VersionInfoXML("${matlab_root}" _matlab_version_tmp)
    if(NOT "${_matlab_version_tmp}" STREQUAL "unknown")
      set(Matlab_VERSION_STRING_INTERNAL ${_matlab_version_tmp} CACHE INTERNAL "Matlab version (automatically determined)")
      set(Matlab_OR_MCR_INTERNAL ${matlab_or_mcr} CACHE INTERNAL "Whether Matlab root contains MATLAB or MCR")
    endif()
  endif() # Matlab or MCR

  # return the updated value
  set(${matlab_final_version} ${Matlab_VERSION_STRING_INTERNAL} PARENT_SCOPE)

endfunction()


function(_Matlab_VersionInfoXML matlab_root _version)

  set(_ver "unknown")

  set(_XMLfile ${matlab_root}/VersionInfo.xml)
  if(EXISTS ${_XMLfile})
    file(READ ${_XMLfile} versioninfo_string)

    # parses "<version>23.2.0.2365128</version>"
    if(versioninfo_string MATCHES "<version>([0-9]+(\\.[0-9]+)+)</version>")
      set(_ver "${CMAKE_MATCH_1}")
    endif()
  endif()

  set(${_version} ${_ver} PARENT_SCOPE)

endfunction()


# Utility function for finding Matlab or MCR on Win32
function(_Matlab_find_instances_win32 matlab_roots)
  # On WIN32, we look for Matlab installation in the registry
  # if unsuccessful, we look for all known revision and filter the existing
  # ones.

  # testing if we are able to extract the needed information from the registry
  set(_matlab_versions_from_registry)

  matlab_extract_all_installed_versions_from_registry(_matlab_versions_from_registry ${ARGN})

  # the returned list is empty, doing the search on all known versions
  if(NOT _matlab_versions_from_registry)
    if(MATLAB_FIND_DEBUG)
      message(STATUS "[MATLAB] Search for Matlab from the registry unsuccessful, testing all supported versions")
    endif()
    extract_matlab_versions_from_registry_brute_force(_matlab_versions_from_registry)
  endif()

  # filtering the results with the registry keys
  matlab_get_all_valid_matlab_roots_from_registry("${_matlab_versions_from_registry}" _matlab_possible_roots ${ARGN})
  set(${matlab_roots} ${_matlab_possible_roots} PARENT_SCOPE)

endfunction()

# Utility function for finding Matlab or MCR on macOS
function(_Matlab_find_instances_macos matlab_roots)

  set(_matlab_possible_roots)
  # on macOS, we look for the standard /Applications paths
  # this corresponds to the behavior on Windows. On Linux, we do not have
  # any other guess.
  matlab_get_supported_releases(_matlab_releases)
  if(MATLAB_FIND_DEBUG)
    message(STATUS "[MATLAB] Matlab supported versions ${_matlab_releases}. If more version should be supported "
                 "the variable MATLAB_ADDITIONAL_VERSIONS can be set according to the documentation")
  endif()

  foreach(_matlab_current_release IN LISTS _matlab_releases)
    foreach(_macos_app_base IN ITEMS "$ENV{HOME}/Applications" "/Applications")
      matlab_get_version_from_release_name("${_matlab_current_release}" _matlab_current_version)
      string(REPLACE "." "" _matlab_current_version_without_dot "${_matlab_current_version}")
      set(_matlab_base_path "${_macos_app_base}/MATLAB_${_matlab_current_release}.app")

      _Matlab_VersionInfoXML("${_matlab_base_path}" _matlab_version_tmp)
      if(NOT "${_matlab_version_tmp}" STREQUAL "unknown")
        set(_matlab_current_version ${_matlab_version_tmp})
      endif()

      # Check Matlab, has precedence over MCR
      if(IS_DIRECTORY "${_matlab_base_path}")
        if(MATLAB_FIND_DEBUG)
          message(STATUS "[MATLAB] Found version ${_matlab_current_release} (${_matlab_current_version}) in ${_matlab_base_path}")
        endif()
        list(APPEND _matlab_possible_roots "MATLAB" ${_matlab_current_version} ${_matlab_base_path})
      endif()

      # Checks MCR
      set(_mcr_path "${_macos_app_base}/MATLAB/MATLAB_Runtime/v${_matlab_current_version_without_dot}")
      if(IS_DIRECTORY "${_mcr_path}")
        if(MATLAB_FIND_DEBUG)
          message(STATUS "[MATLAB] Found MCR version ${_matlab_current_release} (${_matlab_current_version}) in ${_mcr_path}")
        endif()
        list(APPEND _matlab_possible_roots "MCR" ${_matlab_current_version} ${_mcr_path})
      endif()
    endforeach()
  endforeach()
  set(${matlab_roots} ${_matlab_possible_roots} PARENT_SCOPE)

endfunction()

# Utility function for finding Matlab or MCR from the PATH
function(_Matlab_find_instances_from_path matlab_roots)

  set(_matlab_possible_roots)

  # At this point, we have no other choice than trying to find it from PATH.
  # If set by the user, this won't change.
  find_program(
    _matlab_main_tmp
    NAMES matlab)

  if(_matlab_main_tmp)
    # we then populate the list of roots, with empty version
    if(MATLAB_FIND_DEBUG)
      message(STATUS "[MATLAB] matlab found from PATH: ${_matlab_main_tmp}")
    endif()

    # resolve symlinks
    get_filename_component(_matlab_current_location "${_matlab_main_tmp}" REALPATH)

    # get the directory (the command below has to be run twice)
    # this will be the matlab root
    get_filename_component(_matlab_current_location "${_matlab_current_location}" DIRECTORY)
    get_filename_component(_matlab_current_location "${_matlab_current_location}" DIRECTORY) # Matlab should be in bin

    # We found the Matlab program
    list(APPEND _matlab_possible_roots "MATLAB" "NOTFOUND" ${_matlab_current_location})

    # we remove this from the CACHE
    unset(_matlab_main_tmp CACHE)
  else()
    find_program(
      _matlab_mex_tmp
      NAMES mex)
    if(_matlab_mex_tmp)
      # we then populate the list of roots, with empty version
      if(MATLAB_FIND_DEBUG)
        message(STATUS "[MATLAB] mex compiler found from PATH: ${_matlab_mex_tmp}")
      endif()

      # resolve symlinks
      get_filename_component(_mex_current_location "${_matlab_mex_tmp}" REALPATH)

      # get the directory (the command below has to be run twice)
      # this will be the matlab root
      get_filename_component(_mex_current_location "${_mex_current_location}" DIRECTORY)
      get_filename_component(_mex_current_location "${_mex_current_location}" DIRECTORY) # Matlab Runtime mex compiler should be in bin

      # We found the Matlab program
      list(APPEND _matlab_possible_roots "MCR" "NOTFOUND" ${_mex_current_location})

      unset(_matlab_mex_tmp CACHE)
    else()
      if(MATLAB_FIND_DEBUG)
        message(STATUS "[MATLAB] mex compiler not found")
      endif()
    endif()


  endif()

  set(${matlab_roots} ${_matlab_possible_roots} PARENT_SCOPE)
endfunction()


# ###################################
# Exploring the possible Matlab_ROOTS

# this variable will get all Matlab installations found in the current system.
set(_matlab_possible_roots)

if(NOT DEFINED Matlab_ROOT AND DEFINED ENV{Matlab_ROOT})
  set(Matlab_ROOT $ENV{Matlab_ROOT})
endif()
if(DEFINED Matlab_ROOT)
  set(Matlab_ROOT_DIR ${Matlab_ROOT})
endif()

if(Matlab_ROOT_DIR)
  # if the user specifies a possible root, we keep this one

  if(NOT IS_DIRECTORY "${Matlab_ROOT_DIR}")
    # if Matlab_ROOT_DIR specified but erroneous
    if(MATLAB_FIND_DEBUG)
      message(WARNING "[MATLAB] the specified path for Matlab_ROOT_DIR does not exist (${Matlab_ROOT_DIR})")
    endif()
  else()
    if("${Matlab_OR_MCR_INTERNAL}" STREQUAL "")
      set(_matlab_cached_matlab_or_mcr "UNKNOWN")
    else()
      set(_matlab_cached_matlab_or_mcr "${Matlab_OR_MCR_INTERNAL}")
    endif()
    # NOTFOUND indicates the code below to search for the version automatically
    if("${Matlab_VERSION_STRING_INTERNAL}" STREQUAL "")
      set(_matlab_cached_version "NOTFOUND") # empty version, empty MCR/Matlab indication
    else()
      set(_matlab_cached_version "${Matlab_VERSION_STRING_INTERNAL}") # cached version
    endif()
    list(APPEND _matlab_possible_roots "${_matlab_cached_matlab_or_mcr}" "${_matlab_cached_version}" "${Matlab_ROOT_DIR}")
  endif()
else()

  # if the user does not specify the possible installation root, we look for
  # one installation using the appropriate heuristics.
  # There is apparently no standard way on Linux.
  if(CMAKE_HOST_WIN32)
    if(NOT DEFINED Matlab_FIND_REGISTRY_VIEW)
      set(Matlab_FIND_REGISTRY_VIEW TARGET)
    endif()
    _Matlab_find_instances_win32(_matlab_possible_roots_win32 REGISTRY_VIEW ${Matlab_FIND_REGISTRY_VIEW})
    list(APPEND _matlab_possible_roots ${_matlab_possible_roots_win32})
  elseif(APPLE)
    _Matlab_find_instances_macos(_matlab_possible_roots_macos)
    list(APPEND _matlab_possible_roots ${_matlab_possible_roots_macos})
  endif()
endif()


list(LENGTH _matlab_possible_roots _numbers_of_matlab_roots)
if(_numbers_of_matlab_roots EQUAL 0)
  # if we have not found anything, we fall back on the PATH
  _Matlab_find_instances_from_path(_matlab_possible_roots)
endif()


if(MATLAB_FIND_DEBUG)
  message(STATUS "[MATLAB] Matlab root folders are ${_matlab_possible_roots}")
endif()

# take the first possible Matlab root
list(LENGTH _matlab_possible_roots _numbers_of_matlab_roots)
set(Matlab_VERSION_STRING "NOTFOUND")
set(Matlab_Or_MCR "UNKNOWN")
if(_numbers_of_matlab_roots GREATER 0)
  set(_list_index -1)
  foreach(_matlab_root_index RANGE 1 ${_numbers_of_matlab_roots} 3)
    list(GET _matlab_possible_roots ${_matlab_root_index} _matlab_root_version)
    find_package_check_version(${_matlab_root_version} _matlab_version_ok HANDLE_VERSION_RANGE)
    if(_matlab_version_ok)
      set(_list_index ${_matlab_root_index})
      break()
    endif()
  endforeach()

  if(_list_index LESS 0)
    set(_list_index 1)
  endif()

  math(EXPR _matlab_or_mcr_index "${_list_index} - 1")
  math(EXPR _matlab_root_dir_index "${_list_index} + 1")
  list(GET _matlab_possible_roots ${_matlab_or_mcr_index} Matlab_Or_MCR)
  list(GET _matlab_possible_roots ${_list_index} Matlab_VERSION_STRING)
  list(GET _matlab_possible_roots ${_matlab_root_dir_index} Matlab_ROOT_DIR)
  # adding a warning in case of ambiguity
  if(_numbers_of_matlab_roots GREATER 3 AND NOT Matlab_FIND_VERSION_EXACT AND MATLAB_FIND_DEBUG)
    message(WARNING "[MATLAB] Found several distributions of Matlab. Setting the current version to ${Matlab_VERSION_STRING} (located ${Matlab_ROOT_DIR})."
                    " If this is not the desired behavior, use the EXACT keyword or provide the -DMatlab_ROOT_DIR=... on the command line")
  endif()
endif()


# check if the root changed wrt. the previous defined one, if so
# clear all the cached variables for being able to reconfigure properly
if(DEFINED Matlab_ROOT_DIR_LAST_CACHED)

  if(NOT Matlab_ROOT_DIR_LAST_CACHED STREQUAL Matlab_ROOT_DIR)
    set(_Matlab_cached_vars
        Matlab_VERSION_STRING
        Matlab_INCLUDE_DIRS
        Matlab_MEX_LIBRARY
        Matlab_MEX_COMPILER
        Matlab_MCC_COMPILER
        Matlab_MAIN_PROGRAM
        Matlab_MX_LIBRARY
        Matlab_ENG_LIBRARY
        Matlab_MAT_LIBRARY
        Matlab_ENGINE_LIBRARY
        Matlab_DATAARRAY_LIBRARY
        Matlab_MEX_EXTENSION
        Matlab_SIMULINK_INCLUDE_DIR

        # internal
        Matlab_MEXEXTENSIONS_PROG
        Matlab_ROOT_DIR_LAST_CACHED
        #Matlab_PROG_VERSION_STRING_AUTO_DETECT
        #Matlab_VERSION_STRING_INTERNAL
        )
    foreach(_var IN LISTS _Matlab_cached_vars)
      if(DEFINED ${_var})
        unset(${_var} CACHE)
      endif()
    endforeach()
  endif()
endif()

set(Matlab_ROOT_DIR_LAST_CACHED ${Matlab_ROOT_DIR} CACHE INTERNAL "last Matlab root dir location")
set(Matlab_ROOT_DIR ${Matlab_ROOT_DIR} CACHE PATH "Matlab installation root path" FORCE)

# Fix the version, in case this one is NOTFOUND
_Matlab_get_version_from_root(
  "${Matlab_ROOT_DIR}"
  "${Matlab_Or_MCR}"
  ${Matlab_VERSION_STRING}
  Matlab_VERSION_STRING
)

if(MATLAB_FIND_DEBUG)
  message(STATUS "[MATLAB] Current version is ${Matlab_VERSION_STRING} located ${Matlab_ROOT_DIR}")
endif()

# MATLAB 9.4 (R2018a) and newer have a new C++ API
# This API pulls additional required libraries.
if(NOT ${Matlab_VERSION_STRING} VERSION_LESS "9.4")
  set(Matlab_HAS_CPP_API 1)
endif()

if(Matlab_ROOT_DIR)
  file(TO_CMAKE_PATH ${Matlab_ROOT_DIR} Matlab_ROOT_DIR)
endif()


if(NOT DEFINED Matlab_MEX_EXTENSION)
  set(_matlab_mex_extension "")
  matlab_get_mex_suffix("${Matlab_ROOT_DIR}" _matlab_mex_extension)

  # This variable goes to the cache.
  set(Matlab_MEX_EXTENSION ${_matlab_mex_extension} CACHE STRING "Extensions for the mex targets (automatically given by Matlab)")
  unset(_matlab_mex_extension)
endif()

if(APPLE)
  set(_matlab_bin_prefix "mac") # i should be for intel
  set(_matlab_bin_suffix_32bits "i")
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" AND Matlab_MEX_EXTENSION MATCHES "a64$")
    # native Apple Silicon Matlab
    set(_matlab_bin_suffix_64bits "a64")
  else()
    # Intel Mac OR Apple Silicon using Rosetta for Matlab
    set(_matlab_bin_suffix_64bits "i64")
  endif()
elseif(UNIX)
  set(_matlab_bin_prefix "gln")
  set(_matlab_bin_suffix_32bits "x86")
  set(_matlab_bin_suffix_64bits "xa64")
else()
  set(_matlab_bin_prefix "win")
  set(_matlab_bin_suffix_32bits "32")
  set(_matlab_bin_suffix_64bits "64")
endif()



set(MATLAB_INCLUDE_DIR_TO_LOOK ${Matlab_ROOT_DIR}/extern/include)
if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(_matlab_current_suffix ${_matlab_bin_suffix_32bits})
else()
  set(_matlab_current_suffix ${_matlab_bin_suffix_64bits})
endif()

set(Matlab_BINARIES_DIR
    ${Matlab_ROOT_DIR}/bin/${_matlab_bin_prefix}${_matlab_current_suffix})
set(Matlab_EXTERN_LIBRARY_DIR
    ${Matlab_ROOT_DIR}/extern/lib/${_matlab_bin_prefix}${_matlab_current_suffix})
set(Matlab_EXTERN_BINARIES_DIR
    ${Matlab_ROOT_DIR}/extern/bin/${_matlab_bin_prefix}${_matlab_current_suffix})

if(WIN32)
  if(MINGW)
    set(_matlab_lib_dir_for_search ${Matlab_EXTERN_LIBRARY_DIR}/mingw64)
  else()
    set(_matlab_lib_dir_for_search ${Matlab_EXTERN_LIBRARY_DIR}/microsoft)
  endif()
  set(_matlab_lib_prefix_for_search "lib")
else()
  set(_matlab_lib_dir_for_search ${Matlab_BINARIES_DIR} ${Matlab_EXTERN_BINARIES_DIR})
  set(_matlab_lib_prefix_for_search "lib")
endif()


if(MATLAB_FIND_DEBUG)
  message(STATUS "[MATLAB] _matlab_lib_prefix_for_search = ${_matlab_lib_prefix_for_search} | _matlab_lib_dir_for_search = ${_matlab_lib_dir_for_search}")
endif()



# internal
# This small stub around find_library is to prevent any pollution of CMAKE_FIND_LIBRARY_PREFIXES in the global scope.
# This is the function to be used below instead of the find_library directives.
function(_Matlab_find_library _matlab_library_prefix)
  list(APPEND CMAKE_FIND_LIBRARY_PREFIXES ${_matlab_library_prefix})
  find_library(${ARGN})
endfunction()


# the matlab root is required
set(_matlab_required_variables Matlab_ROOT_DIR)
set(Matlab_LIBRARIES)

# Order is as follow:
# - unconditionally required libraries/headers first
# - then library components
# - then program components

# the MEX library/header are required
find_path(
  Matlab_INCLUDE_DIRS
  NAMES mex.h matrix.h
  PATHS ${MATLAB_INCLUDE_DIR_TO_LOOK}
  NO_DEFAULT_PATH
  )
list(APPEND _matlab_required_variables Matlab_INCLUDE_DIRS)

_Matlab_find_library(
  ${_matlab_lib_prefix_for_search}
  Matlab_MEX_LIBRARY
  NAMES mex
  PATHS ${_matlab_lib_dir_for_search}
  NO_DEFAULT_PATH
)
if(Matlab_MEX_LIBRARY)
  set(Matlab_MEX_LIBRARY_FOUND TRUE)
  list(APPEND Matlab_LIBRARIES ${Matlab_MEX_LIBRARY})
endif()
if(MATLAB_FIND_DEBUG)
  message(STATUS "[MATLAB] mex C library: ${Matlab_MEX_LIBRARY}")
endif()

# The MX library is required
_Matlab_find_library(
  ${_matlab_lib_prefix_for_search}
  Matlab_MX_LIBRARY
  NAMES mx
  PATHS ${_matlab_lib_dir_for_search}
  NO_DEFAULT_PATH
)
if(Matlab_MX_LIBRARY)
  set(Matlab_MX_LIBRARY_FOUND TRUE)
  list(APPEND Matlab_LIBRARIES ${Matlab_MX_LIBRARY})
endif()
if(MATLAB_FIND_DEBUG)
  message(STATUS "[MATLAB] mx C library: ${Matlab_MX_LIBRARY}")
endif()

if(Matlab_Or_MCR STREQUAL "MATLAB" OR Matlab_Or_MCR STREQUAL "UNKNOWN")
  list(APPEND _matlab_required_variables Matlab_MEX_LIBRARY)

  # the MEX extension is required
  list(APPEND _matlab_required_variables Matlab_MEX_EXTENSION)

  list(APPEND _matlab_required_variables Matlab_MX_LIBRARY)
endif()

if(Matlab_HAS_CPP_API)

  # The MatlabEngine library is required for R2018a+
  _Matlab_find_library(
    ${_matlab_lib_prefix_for_search}
    Matlab_ENGINE_LIBRARY
    NAMES MatlabEngine
    PATHS ${_matlab_lib_dir_for_search}
    DOC "MatlabEngine Library"
    NO_DEFAULT_PATH
  )
  if(Matlab_ENGINE_LIBRARY)
    set(Matlab_ENGINE_LIBRARY_FOUND TRUE)
    list(APPEND Matlab_LIBRARIES ${Matlab_ENGINE_LIBRARY})
  endif()
  if(MATLAB_FIND_DEBUG)
    message(STATUS "[MATLAB] Engine C++ library: ${Matlab_ENGINE_LIBRARY}")
  endif()

  # The MatlabDataArray library is required for R2018a+
  _Matlab_find_library(
    ${_matlab_lib_prefix_for_search}
    Matlab_DATAARRAY_LIBRARY
    NAMES MatlabDataArray
    PATHS ${_matlab_lib_dir_for_search}
    DOC "MatlabDataArray Library"
    NO_DEFAULT_PATH
  )
  if(Matlab_DATAARRAY_LIBRARY)
    set(Matlab_DATAARRAY_LIBRARY_FOUND TRUE)
    list(APPEND Matlab_LIBRARIES ${Matlab_DATAARRAY_LIBRARY})
  endif()
  if(MATLAB_FIND_DEBUG)
    message(STATUS "[MATLAB] Data array C++ library: ${Matlab_DATAARRAY_LIBRARY}")
  endif()

endif()

# Component ENG library
if("ENG_LIBRARY" IN_LIST Matlab_FIND_COMPONENTS)
  _Matlab_find_library(
    ${_matlab_lib_prefix_for_search}
    Matlab_ENG_LIBRARY
    NAMES eng
    PATHS ${_matlab_lib_dir_for_search}
    NO_DEFAULT_PATH
  )
  if(Matlab_ENG_LIBRARY)
    set(Matlab_ENG_LIBRARY_FOUND TRUE)
    list(APPEND Matlab_LIBRARIES ${Matlab_ENG_LIBRARY})
  endif()
  if(MATLAB_FIND_DEBUG)
    message(STATUS "[MATLAB] eng C library: ${Matlab_ENG_LIBRARY}")
  endif()
endif()

# Component MAT library
if("MAT_LIBRARY" IN_LIST Matlab_FIND_COMPONENTS)
  _Matlab_find_library(
    ${_matlab_lib_prefix_for_search}
    Matlab_MAT_LIBRARY
    NAMES mat
    PATHS ${_matlab_lib_dir_for_search}
    NO_DEFAULT_PATH
  )
  if(Matlab_MAT_LIBRARY)
    set(Matlab_MAT_LIBRARY_FOUND TRUE)
    list(APPEND Matlab_LIBRARIES ${Matlab_MAT_LIBRARY})
  endif()
  if(MATLAB_FIND_DEBUG)
    message(STATUS "[MATLAB] mat C library: ${Matlab_MAT_LIBRARY}")
  endif()
endif()

# Component Simulink
if("SIMULINK" IN_LIST Matlab_FIND_COMPONENTS)
  find_path(
    Matlab_SIMULINK_INCLUDE_DIR
    NAMES simstruc.h
    PATHS "${Matlab_ROOT_DIR}/simulink/include"
    NO_DEFAULT_PATH
    )
  if(Matlab_SIMULINK_INCLUDE_DIR)
    set(Matlab_SIMULINK_FOUND TRUE)
    list(APPEND Matlab_INCLUDE_DIRS "${Matlab_SIMULINK_INCLUDE_DIR}")
  endif()
  if(MATLAB_FIND_DEBUG)
    message(STATUS "[MATLAB] Simulink include dir: ${Matlab_SIMULINK_INCLUDE_DIR}")
  endif()
endif()

# component Matlab program
if("MAIN_PROGRAM" IN_LIST Matlab_FIND_COMPONENTS)
  find_program(
    Matlab_MAIN_PROGRAM
    NAMES matlab
    PATHS ${Matlab_ROOT_DIR} ${Matlab_ROOT_DIR}/bin
    DOC "Matlab main program"
    NO_DEFAULT_PATH
  )
  if(Matlab_MAIN_PROGRAM)
    set(Matlab_MAIN_PROGRAM_FOUND TRUE)
  endif()
  if(MATLAB_FIND_DEBUG)
    message(STATUS "[MATLAB] Main program: ${Matlab_MAIN_PROGRAM}")
  endif()
endif()

# component Mex Compiler
if("MEX_COMPILER" IN_LIST Matlab_FIND_COMPONENTS)
  find_program(
    Matlab_MEX_COMPILER
    NAMES "mex"
    PATHS ${Matlab_BINARIES_DIR}
    DOC "Matlab MEX compiler"
    NO_DEFAULT_PATH
  )
  if(Matlab_MEX_COMPILER)
    set(Matlab_MEX_COMPILER_FOUND TRUE)
  endif()
  if(MATLAB_FIND_DEBUG)
    message(STATUS "[MATLAB] MEX compiler: ${Matlab_MEX_COMPILER}")
  endif()
endif()

# component MCC Compiler
if("MCC_COMPILER" IN_LIST Matlab_FIND_COMPONENTS)
  find_program(
    Matlab_MCC_COMPILER
    NAMES "mcc"
    PATHS ${Matlab_BINARIES_DIR}
    DOC "Matlab MCC compiler"
    NO_DEFAULT_PATH
  )
  if(Matlab_MCC_COMPILER)
    set(Matlab_MCC_COMPILER_FOUND TRUE)
  endif()
  if(MATLAB_FIND_DEBUG)
    message(STATUS "[MATLAB] MCC compiler: ${Matlab_MCC_COMPILER}")
  endif()
endif()

# internal
# This small stub permits to add imported targets for the found MATLAB libraries
function(_Matlab_add_imported_target _matlab_library_variable_name _matlab_library_target_name)
  if(Matlab_${_matlab_library_variable_name}_LIBRARY)
    if(NOT TARGET Matlab::${_matlab_library_target_name})
      add_library(Matlab::${_matlab_library_target_name} UNKNOWN IMPORTED)
      set_target_properties(Matlab::${_matlab_library_target_name} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Matlab_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${Matlab_${_matlab_library_variable_name}_LIBRARY}")
      if(_matlab_library_target_name STREQUAL "mex" OR
         _matlab_library_target_name STREQUAL "eng" OR
         _matlab_library_target_name STREQUAL "mat")
        set_target_properties(Matlab::${_matlab_library_target_name} PROPERTIES
          INTERFACE_LINK_LIBRARIES Matlab::mx)
      endif()
    endif()
  endif()
endfunction()

_Matlab_add_imported_target(MX mx)
_Matlab_add_imported_target(MEX mex)
_Matlab_add_imported_target(ENG eng)
_Matlab_add_imported_target(MAT mat)
_Matlab_add_imported_target(ENGINE MatlabEngine)
_Matlab_add_imported_target(DATAARRAY MatlabDataArray)

set(Matlab_VERSION ${Matlab_VERSION_STRING})

find_package_handle_standard_args(
  Matlab
  REQUIRED_VARS ${_matlab_required_variables}
  VERSION_VAR Matlab_VERSION
  HANDLE_VERSION_RANGE
  HANDLE_COMPONENTS)

unset(_matlab_required_variables)
unset(_matlab_bin_prefix)
unset(_matlab_bin_suffix_32bits)
unset(_matlab_bin_suffix_64bits)
unset(_matlab_current_suffix)
unset(_matlab_lib_dir_for_search)
unset(_matlab_lib_prefix_for_search)

if(Matlab_INCLUDE_DIRS AND Matlab_LIBRARIES)
  mark_as_advanced(
    Matlab_MEX_LIBRARY
    Matlab_MX_LIBRARY
    Matlab_ENG_LIBRARY
    Matlab_ENGINE_LIBRARY
    Matlab_DATAARRAY_LIBRARY
    Matlab_MAT_LIBRARY
    Matlab_INCLUDE_DIRS
    Matlab_FOUND
    Matlab_MAIN_PROGRAM
    Matlab_MEXEXTENSIONS_PROG
    Matlab_MEX_EXTENSION
  )
endif()

cmake_policy(POP)
