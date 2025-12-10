# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CPackIFW
--------

.. versionadded:: 3.1

This module looks for the location of the command-line utilities supplied with the
`Qt Installer Framework <https://doc.qt.io/qtinstallerframework/index.html>`_
(QtIFW).

Load this module in a CMake project with:

.. code-block:: cmake

  include(CPackIFW)

The module also defines several commands to control the behavior of the
:cpack_gen:`CPack IFW Generator`.

Commands
^^^^^^^^

The module defines the following commands:

.. command:: cpack_ifw_configure_component

  Sets the arguments specific to the CPack IFW generator.

  .. code-block:: cmake

    cpack_ifw_configure_component(<compname> [COMMON] [ESSENTIAL] [VIRTUAL]
                        [FORCED_INSTALLATION] [REQUIRES_ADMIN_RIGHTS]
                        [NAME <name>]
                        [DISPLAY_NAME <display_name>] # Note: Internationalization supported
                        [DESCRIPTION <description>] # Note: Internationalization supported
                        [UPDATE_TEXT <update_text>]
                        [VERSION <version>]
                        [RELEASE_DATE <release_date>]
                        [SCRIPT <script>]
                        [PRIORITY|SORTING_PRIORITY <sorting_priority>] # Note: PRIORITY is deprecated
                        [DEPENDS|DEPENDENCIES <com_id> ...]
                        [AUTO_DEPEND_ON <comp_id> ...]
                        [LICENSES <display_name> <file_path> ...]
                        [DEFAULT <value>]
                        [USER_INTERFACES <file_path> <file_path> ...]
                        [TRANSLATIONS <file_path> <file_path> ...]
                        [REPLACES <comp_id> ...]
                        [CHECKABLE <value>])

  This command should be called after :command:`cpack_add_component` command.

  ``COMMON``
    if set, then the component will be packaged and installed as part
    of a group to which it belongs.

  ``ESSENTIAL``
    .. versionadded:: 3.6

    if set, then the package manager stays disabled until that
    component is updated.

  ``VIRTUAL``
    .. versionadded:: 3.8

    if set, then the component will be hidden from the installer.
    It is a equivalent of the ``HIDDEN`` option from the
    :command:`cpack_add_component` command.

  ``FORCED_INSTALLATION``
    .. versionadded:: 3.8

    if set, then the component must always be installed.
    It is a equivalent of the ``REQUIRED`` option from the
    :command:`cpack_add_component` command.

  ``REQUIRES_ADMIN_RIGHTS``
    .. versionadded:: 3.8

    set it if the component needs to be installed with elevated permissions.

  ``NAME``
    is used to create domain-like identification for this component.
    By default used origin component name.

  ``DISPLAY_NAME``
    .. versionadded:: 3.8

    set to rewrite original name configured by
    :command:`cpack_add_component` command.

  ``DESCRIPTION``
    .. versionadded:: 3.8

    set to rewrite original description configured by
    :command:`cpack_add_component` command.

  ``UPDATE_TEXT``
    .. versionadded:: 3.8

    will be added to the component description if this is an update to
    the component.

  ``VERSION``
    is version of component.
    By default used :variable:`CPACK_PACKAGE_VERSION`.

  ``RELEASE_DATE``
    .. versionadded:: 3.8

    keep empty to auto generate.

  ``SCRIPT``
    is a relative or absolute path to operations script
    for this component.

  ``SORTING_PRIORITY``
    .. versionadded:: 3.8

    is priority of the component in the tree.

  ``PRIORITY``
    .. deprecated:: 3.8
      Old name for ``SORTING_PRIORITY``.

  ``DEPENDS``, ``DEPENDENCIES``
    .. versionadded:: 3.8

    list of dependency component or component group identifiers in
    QtIFW style.

    .. versionadded:: 3.21

    Component or group names listed as dependencies may contain hyphens.
    This requires QtIFW 3.1 or later.

  ``AUTO_DEPEND_ON``
    .. versionadded:: 3.8

    list of identifiers of component or component group in QtIFW style
    that this component has an automatic dependency on.

  ``LICENSES``
    pair of <display_name> and <file_path> of license text for this
    component. You can specify more then one license.

  ``DEFAULT``
    .. versionadded:: 3.8

    Possible values are: TRUE, FALSE, and SCRIPT.
    Set to FALSE to disable the component in the installer or to SCRIPT
    to resolved during runtime (don't forget add the file of the script
    as a value of the ``SCRIPT`` option).

  ``USER_INTERFACES``
    .. versionadded:: 3.7

    is a list of <file_path> ('.ui' files) representing pages to load.

  ``TRANSLATIONS``
    .. versionadded:: 3.8

    is a list of <file_path> ('.qm' files) representing translations to load.

  ``REPLACES``
    .. versionadded:: 3.10

    list of identifiers of component or component group to replace.

  ``CHECKABLE``
    .. versionadded:: 3.10

    Possible values are: TRUE, FALSE.
    Set to FALSE if you want to hide the checkbox for an item.
    This is useful when only a few subcomponents should be selected
    instead of all.


.. command:: cpack_ifw_configure_component_group

  Sets the arguments specific to the CPack IFW generator.

  .. code-block:: cmake

    cpack_ifw_configure_component_group(<groupname> [VIRTUAL]
                        [FORCED_INSTALLATION] [REQUIRES_ADMIN_RIGHTS]
                        [NAME <name>]
                        [DISPLAY_NAME <display_name>] # Note: Internationalization supported
                        [DESCRIPTION <description>] # Note: Internationalization supported
                        [UPDATE_TEXT <update_text>]
                        [VERSION <version>]
                        [RELEASE_DATE <release_date>]
                        [SCRIPT <script>]
                        [PRIORITY|SORTING_PRIORITY <sorting_priority>] # Note: PRIORITY is deprecated
                        [DEPENDS|DEPENDENCIES <com_id> ...]
                        [AUTO_DEPEND_ON <comp_id> ...]
                        [LICENSES <display_name> <file_path> ...]
                        [DEFAULT <value>]
                        [USER_INTERFACES <file_path> <file_path> ...]
                        [TRANSLATIONS <file_path> <file_path> ...]
                        [REPLACES <comp_id> ...]
                        [CHECKABLE <value>])

  This command should be called after :command:`cpack_add_component_group`
  command.

  ``VIRTUAL``
    .. versionadded:: 3.8

    if set, then the group will be hidden from the installer.
    Note that setting this on a root component does not work.

  ``FORCED_INSTALLATION``
    .. versionadded:: 3.8

    if set, then the group must always be installed.

  ``REQUIRES_ADMIN_RIGHTS``
    .. versionadded:: 3.8

    set it if the component group needs to be installed with elevated
    permissions.

  ``NAME``
    is used to create domain-like identification for this component group.
    By default used origin component group name.

  ``DISPLAY_NAME``
    .. versionadded:: 3.8

    set to rewrite original name configured by
    :command:`cpack_add_component_group` command.

  ``DESCRIPTION``
    .. versionadded:: 3.8

    set to rewrite original description configured by
    :command:`cpack_add_component_group` command.

  ``UPDATE_TEXT``
    .. versionadded:: 3.8

    will be added to the component group description if this is an update to
    the component group.

  ``VERSION``
    is version of component group.
    By default used :variable:`CPACK_PACKAGE_VERSION`.

  ``RELEASE_DATE``
    .. versionadded:: 3.8

    keep empty to auto generate.

  ``SCRIPT``
    is a relative or absolute path to operations script
    for this component group.

  ``SORTING_PRIORITY``
    is priority of the component group in the tree.

  ``PRIORITY``
    .. deprecated:: 3.8
      Old name for ``SORTING_PRIORITY``.

  ``DEPENDS``, ``DEPENDENCIES``
    .. versionadded:: 3.8

    list of dependency component or component group identifiers in
    QtIFW style.

    .. versionadded:: 3.21

    Component or group names listed as dependencies may contain hyphens.
    This requires QtIFW 3.1 or later.

  ``AUTO_DEPEND_ON``
    .. versionadded:: 3.8

    list of identifiers of component or component group in QtIFW style
    that this component group has an automatic dependency on.

  ``LICENSES``
    pair of <display_name> and <file_path> of license text for this
    component group. You can specify more then one license.

  ``DEFAULT``
    .. versionadded:: 3.8

    Possible values are: TRUE, FALSE, and SCRIPT.
    Set to TRUE to preselect the group in the installer
    (this takes effect only on groups that have no visible child components)
    or to SCRIPT to resolved during runtime (don't forget add the file of
    the script as a value of the ``SCRIPT`` option).

  ``USER_INTERFACES``
    .. versionadded:: 3.7

    is a list of <file_path> ('.ui' files) representing pages to load.

  ``TRANSLATIONS``
    .. versionadded:: 3.8

    is a list of <file_path> ('.qm' files) representing translations to load.

  ``REPLACES``
    .. versionadded:: 3.10

    list of identifiers of component or component group to replace.

  ``CHECKABLE``
    .. versionadded:: 3.10

    Possible values are: TRUE, FALSE.
    Set to FALSE if you want to hide the checkbox for an item.
    This is useful when only a few subcomponents should be selected
    instead of all.


.. command:: cpack_ifw_add_repository

  Add QtIFW specific remote repository to binary installer.

  .. code-block:: cmake

    cpack_ifw_add_repository(<reponame> [DISABLED]
                        URL <url>
                        [USERNAME <username>]
                        [PASSWORD <password>]
                        [DISPLAY_NAME <display_name>])

  This command will also add the <reponame> repository
  to a variable :variable:`CPACK_IFW_REPOSITORIES_ALL`.

  ``DISABLED``
    if set, then the repository will be disabled by default.

  ``URL``
    is points to a list of available components.

  ``USERNAME``
    is used as user on a protected repository.

  ``PASSWORD``
    is password to use on a protected repository.

  ``DISPLAY_NAME``
    is string to display instead of the URL.


.. command:: cpack_ifw_update_repository

  .. versionadded:: 3.6

  Update QtIFW specific repository from remote repository.

  .. code-block:: cmake

    cpack_ifw_update_repository(<reponame>
                        [[ADD|REMOVE] URL <url>]|
                         [REPLACE OLD_URL <old_url> NEW_URL <new_url>]]
                        [USERNAME <username>]
                        [PASSWORD <password>]
                        [DISPLAY_NAME <display_name>])

  This command will also add the <reponame> repository
  to a variable :variable:`CPACK_IFW_REPOSITORIES_ALL`.

  ``URL``
    is points to a list of available components.

  ``OLD_URL``
    is points to a list that will replaced.

  ``NEW_URL``
    is points to a list that will replace to.

  ``USERNAME``
    is used as user on a protected repository.

  ``PASSWORD``
    is password to use on a protected repository.

  ``DISPLAY_NAME``
    is string to display instead of the URL.


.. command:: cpack_ifw_add_package_resources

  .. versionadded:: 3.7

  Add additional resources in the installer binary.

  .. code-block:: cmake

    cpack_ifw_add_package_resources(<file_path> <file_path> ...)

  This command will also add the specified files
  to a variable :variable:`CPACK_IFW_PACKAGE_RESOURCES`.

#]=======================================================================]

# TODO:
# All of the internal implementation CMake modules for other CPack generators
# have been moved into the Internal/CPack directory. This one has not, because
# it contains user-facing macros which would be lost if it were moved. At some
# point, this module should be split into user-facing macros (which would live
# in this module) and internal implementation details (which would live in
# Internal/CPack/CPackIFW.cmake).

#=============================================================================
# Search Qt Installer Framework tools
#=============================================================================

# Default path

foreach(_CPACK_IFW_PATH_VAR "CPACK_IFW_ROOT" "QTIFWDIR" "QTDIR")
  if(DEFINED ${_CPACK_IFW_PATH_VAR}
    AND NOT "${${_CPACK_IFW_PATH_VAR}}" STREQUAL "")
    list(APPEND _CPACK_IFW_PATHS "${${_CPACK_IFW_PATH_VAR}}")
  endif()
  if(NOT "$ENV{${_CPACK_IFW_PATH_VAR}}" STREQUAL "")
    list(APPEND _CPACK_IFW_PATHS "$ENV{${_CPACK_IFW_PATH_VAR}}")
  endif()
endforeach()
if(WIN32)
  list(APPEND _CPACK_IFW_PATHS
    "$ENV{HOMEDRIVE}/Qt"
    "C:/Qt")
else()
  list(APPEND _CPACK_IFW_PATHS
    "$ENV{HOME}/Qt"
    "/opt/Qt")
endif()
list(REMOVE_DUPLICATES _CPACK_IFW_PATHS)

set(_CPACK_IFW_PREFIXES
  # QtSDK
  "Tools/QtInstallerFramework/"
  # Second branch
  "QtIFW"
  # First branch
  "QtIFW-")

set(_CPACK_IFW_VERSIONS
  "4.5.0"
  "4.5"
  "4.4.2"
  "4.4.1"
  "4.4.0"
  "4.4"
  "4.3.0"
  "4.3"
  "4.2.0"
  "4.2"
  "4.1.1"
  "4.1.0"
  "4.1"
  "4.0.1"
  "4.0.0"
  "4.0"
  "3.2.3"
  "3.2.2"
  "3.2.1"
  "3.2.0"
  "3.2"
  "3.1.1"
  "3.1.0"
  "3.1"
  "3.0.6"
  "3.0.4"
  "3.0.3"
  "3.0.2"
  "3.0.1"
  "3.0.0"
  "3.0"
  "2.3.0"
  "2.3"
  "2.2.0"
  "2.2"
  "2.1.0"
  "2.1"
  "2.0.5"
  "2.0.3"
  "2.0.2"
  "2.0.1"
  "2.0.0"
  "2.0"
  "1.6.0"
  "1.6"
  "1.5.0"
  "1.5"
  "1.4.0"
  "1.4"
  "1.3.0"
  "1.3")

set(_CPACK_IFW_SUFFIXES "bin")
foreach(_CPACK_IFW_PREFIX ${_CPACK_IFW_PREFIXES})
  foreach(_CPACK_IFW_VERSION ${_CPACK_IFW_VERSIONS})
    list(APPEND
      _CPACK_IFW_SUFFIXES "${_CPACK_IFW_PREFIX}${_CPACK_IFW_VERSION}/bin")
  endforeach()
endforeach()

# Look for 'binarycreator'

find_program(CPACK_IFW_BINARYCREATOR_EXECUTABLE
  NAMES binarycreator
  PATHS ${_CPACK_IFW_PATHS}
  PATH_SUFFIXES ${_CPACK_IFW_SUFFIXES}
  DOC "QtIFW binarycreator command line client")

mark_as_advanced(CPACK_IFW_BINARYCREATOR_EXECUTABLE)

# Look for 'repogen'

find_program(CPACK_IFW_REPOGEN_EXECUTABLE
  NAMES repogen
  PATHS ${_CPACK_IFW_PATHS}
  PATH_SUFFIXES ${_CPACK_IFW_SUFFIXES}
  DOC "QtIFW repogen command line client"
  )
mark_as_advanced(CPACK_IFW_REPOGEN_EXECUTABLE)

# Look for 'installerbase'

find_program(CPACK_IFW_INSTALLERBASE_EXECUTABLE
  NAMES installerbase
  PATHS ${_CPACK_IFW_PATHS}
  PATH_SUFFIXES ${_CPACK_IFW_SUFFIXES}
  DOC "QtIFW installer executable base"
  )
mark_as_advanced(CPACK_IFW_INSTALLERBASE_EXECUTABLE)

# Look for 'devtool' (appeared in the second branch)

find_program(CPACK_IFW_DEVTOOL_EXECUTABLE
  NAMES devtool
  PATHS ${_CPACK_IFW_PATHS}
  PATH_SUFFIXES ${_CPACK_IFW_SUFFIXES}
  DOC "QtIFW devtool command line client"
  )
mark_as_advanced(CPACK_IFW_DEVTOOL_EXECUTABLE)

# Look for 'archivegen'

find_program(CPACK_IFW_ARCHIVEGEN_EXECUTABLE
  NAMES archivegen
  PATHS ${_CPACK_IFW_PATHS}
  PATH_SUFFIXES ${_CPACK_IFW_SUFFIXES}
  DOC "QtIFW archivegen command line client"
  )
mark_as_advanced(CPACK_IFW_ARCHIVEGEN_EXECUTABLE)

#
## Next code is included only once
#

if(NOT CPackIFW_CMake_INCLUDED)
set(CPackIFW_CMake_INCLUDED 1)

#=============================================================================
# Framework version
#=============================================================================

set(CPACK_IFW_FRAMEWORK_VERSION_FORCED ""
  CACHE STRING "The forced version of used QtIFW tools")
mark_as_advanced(CPACK_IFW_FRAMEWORK_VERSION_FORCED)
set(CPACK_IFW_FRAMEWORK_VERSION_TIMEOUT 1
  CACHE STRING "The timeout to return QtIFW framework version string from \"installerbase\" executable")
mark_as_advanced(CPACK_IFW_FRAMEWORK_VERSION_TIMEOUT)
if(CPACK_IFW_INSTALLERBASE_EXECUTABLE AND NOT CPACK_IFW_FRAMEWORK_VERSION_FORCED)
  set(CPACK_IFW_FRAMEWORK_VERSION)
  # Invoke version from "installerbase" executable
  foreach(_ifw_version_argument --version --framework-version)
    if(NOT CPACK_IFW_FRAMEWORK_VERSION)
      execute_process(COMMAND
        "${CPACK_IFW_INSTALLERBASE_EXECUTABLE}" ${_ifw_version_argument}
        TIMEOUT ${CPACK_IFW_FRAMEWORK_VERSION_TIMEOUT}
        RESULT_VARIABLE CPACK_IFW_FRAMEWORK_VERSION_RESULT
        OUTPUT_VARIABLE CPACK_IFW_FRAMEWORK_VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ENCODING UTF8)
      if(NOT CPACK_IFW_FRAMEWORK_VERSION_RESULT AND CPACK_IFW_FRAMEWORK_VERSION_OUTPUT)
        string(REGEX MATCH "[0-9]+(\\.[0-9]+)*"
          CPACK_IFW_FRAMEWORK_VERSION "${CPACK_IFW_FRAMEWORK_VERSION_OUTPUT}")
        if(CPACK_IFW_FRAMEWORK_VERSION)
          if("${_ifw_version_argument}" STREQUAL "--framework-version")
            set(CPACK_IFW_FRAMEWORK_VERSION_SOURCE "INSTALLERBASE_FRAMEWORK_VERSION")
          elseif("${_ifw_version_argument}" STREQUAL "--version")
            set(CPACK_IFW_FRAMEWORK_VERSION_SOURCE "INSTALLERBASE_FRAMEWORK_VERSION")
          endif()
        endif()
      endif()
    endif()
  endforeach()
  # Finally try to get version from executable path
  if(NOT CPACK_IFW_FRAMEWORK_VERSION)
    string(REGEX MATCH "[0-9]+(\\.[0-9]+)*"
      CPACK_IFW_FRAMEWORK_VERSION "${CPACK_IFW_INSTALLERBASE_EXECUTABLE}")
    if(CPACK_IFW_FRAMEWORK_VERSION)
      set(CPACK_IFW_FRAMEWORK_VERSION_SOURCE "INSTALLERBASE_PATH")
    endif()
  endif()
elseif(CPACK_IFW_FRAMEWORK_VERSION_FORCED)
  set(CPACK_IFW_FRAMEWORK_VERSION ${CPACK_IFW_FRAMEWORK_VERSION_FORCED})
  set(CPACK_IFW_FRAMEWORK_VERSION_SOURCE "FORCED")
endif()
if(CPACK_IFW_VERBOSE)
  if(CPACK_IFW_FRAMEWORK_VERSION AND CPACK_IFW_FRAMEWORK_VERSION_FORCED)
    message(STATUS "Found QtIFW ${CPACK_IFW_FRAMEWORK_VERSION} (forced) version")
  elseif(CPACK_IFW_FRAMEWORK_VERSION)
    message(STATUS "Found QtIFW ${CPACK_IFW_FRAMEWORK_VERSION} version")
  endif()
endif()
if(CPACK_IFW_INSTALLERBASE_EXECUTABLE AND NOT CPACK_IFW_FRAMEWORK_VERSION)
  message(WARNING "Could not detect QtIFW tools version. Set used version to variable \"CPACK_IFW_FRAMEWORK_VERSION_FORCED\" manually.")
endif()

#=============================================================================
# Macro definition
#=============================================================================

# Macro definition based on CPackComponent

if(NOT CPackComponent_CMake_INCLUDED)
    include(CPackComponent)
endif()

# Resolve full filename for script file
macro(_cpack_ifw_resolve_script _variable)
  set(_ifw_script_macro ${_variable})
  set(_ifw_script_file ${${_ifw_script_macro}})
  if(DEFINED ${_ifw_script_macro})
    get_filename_component(${_ifw_script_macro} ${_ifw_script_file} ABSOLUTE)
    set(_ifw_script_file ${${_ifw_script_macro}})
    if(NOT EXISTS ${_ifw_script_file})
      message(WARNING "CPack IFW: script file \"${_ifw_script_file}\" does not exist")
      set(${_ifw_script_macro})
    endif()
  endif()
endmacro()

# Resolve full path to license file
macro(_cpack_ifw_resolve_licenses _variable)
  if(${_variable})
    set(_ifw_license_file FALSE)
    set(_ifw_licenses_fix)
    foreach(_ifw_licenses_arg ${${_variable}})
      if(_ifw_license_file)
        get_filename_component(_ifw_licenses_arg "${_ifw_licenses_arg}" ABSOLUTE)
        set(_ifw_license_file FALSE)
      else()
        set(_ifw_license_file TRUE)
      endif()
      list(APPEND _ifw_licenses_fix "${_ifw_licenses_arg}")
    endforeach()
    set(${_variable} "${_ifw_licenses_fix}")
  endif()
endmacro()

# Resolve full path to a list of provided files
macro(_cpack_ifw_resolve_file_list _variable)
  if(${_variable})
    set(_ifw_list_fix)
    foreach(_ifw_file_arg ${${_variable}})
      get_filename_component(_ifw_file_arg "${_ifw_file_arg}" ABSOLUTE)
      if(EXISTS ${_ifw_file_arg})
        list(APPEND _ifw_list_fix "${_ifw_file_arg}")
      else()
        message(WARNING "CPack IFW: page file \"${_ifw_file_arg}\" does not exist. Skipping")
      endif()
    endforeach()
    set(${_variable} "${_ifw_list_fix}")
  endif()
endmacro()

# Macro for configure component
macro(cpack_ifw_configure_component compname)

  string(TOUPPER ${compname} _CPACK_IFWCOMP_UNAME)

  set(_IFW_OPT COMMON ESSENTIAL VIRTUAL FORCED_INSTALLATION REQUIRES_ADMIN_RIGHTS)
  set(_IFW_ARGS NAME VERSION RELEASE_DATE SCRIPT PRIORITY SORTING_PRIORITY UPDATE_TEXT DEFAULT CHECKABLE)
  set(_IFW_MULTI_ARGS DISPLAY_NAME DESCRIPTION DEPENDS DEPENDENCIES AUTO_DEPEND_ON LICENSES USER_INTERFACES TRANSLATIONS REPLACES)
  cmake_parse_arguments(CPACK_IFW_COMPONENT_${_CPACK_IFWCOMP_UNAME} "${_IFW_OPT}" "${_IFW_ARGS}" "${_IFW_MULTI_ARGS}" ${ARGN})

  _cpack_ifw_resolve_script(CPACK_IFW_COMPONENT_${_CPACK_IFWCOMP_UNAME}_SCRIPT)
  _cpack_ifw_resolve_licenses(CPACK_IFW_COMPONENT_${_CPACK_IFWCOMP_UNAME}_LICENSES)
  _cpack_ifw_resolve_file_list(CPACK_IFW_COMPONENT_${_CPACK_IFWCOMP_UNAME}_USER_INTERFACES)
  _cpack_ifw_resolve_file_list(CPACK_IFW_COMPONENT_${_CPACK_IFWCOMP_UNAME}_TRANSLATIONS)

  set(_CPACK_IFWCOMP_STR "\n# Configuration for IFW component \"${compname}\"\n")

  foreach(_IFW_ARG_NAME ${_IFW_OPT})
  cpack_append_option_set_command(
    CPACK_IFW_COMPONENT_${_CPACK_IFWCOMP_UNAME}_${_IFW_ARG_NAME}
    _CPACK_IFWCOMP_STR)
  endforeach()

  foreach(_IFW_ARG_NAME ${_IFW_ARGS})
  cpack_append_string_variable_set_command(
    CPACK_IFW_COMPONENT_${_CPACK_IFWCOMP_UNAME}_${_IFW_ARG_NAME}
    _CPACK_IFWCOMP_STR)
  endforeach()

  foreach(_IFW_ARG_NAME ${_IFW_MULTI_ARGS})
  cpack_append_list_variable_set_command(
    CPACK_IFW_COMPONENT_${_CPACK_IFWCOMP_UNAME}_${_IFW_ARG_NAME}
    _CPACK_IFWCOMP_STR)
  endforeach()

  if(CPack_CMake_INCLUDED)
    file(APPEND "${CPACK_OUTPUT_CONFIG_FILE}" "${_CPACK_IFWCOMP_STR}")
  endif()

endmacro()

# Macro for configure group
macro(cpack_ifw_configure_component_group grpname)

  string(TOUPPER ${grpname} _CPACK_IFWGRP_UNAME)

  set(_IFW_OPT VIRTUAL FORCED_INSTALLATION REQUIRES_ADMIN_RIGHTS)
  set(_IFW_ARGS NAME VERSION RELEASE_DATE SCRIPT PRIORITY SORTING_PRIORITY UPDATE_TEXT DEFAULT CHECKABLE)
  set(_IFW_MULTI_ARGS DISPLAY_NAME DESCRIPTION DEPENDS DEPENDENCIES AUTO_DEPEND_ON LICENSES USER_INTERFACES TRANSLATIONS REPLACES)
  cmake_parse_arguments(CPACK_IFW_COMPONENT_GROUP_${_CPACK_IFWGRP_UNAME} "${_IFW_OPT}" "${_IFW_ARGS}" "${_IFW_MULTI_ARGS}" ${ARGN})

  _cpack_ifw_resolve_script(CPACK_IFW_COMPONENT_GROUP_${_CPACK_IFWGRP_UNAME}_SCRIPT)
  _cpack_ifw_resolve_licenses(CPACK_IFW_COMPONENT_GROUP_${_CPACK_IFWGRP_UNAME}_LICENSES)
  _cpack_ifw_resolve_file_list(CPACK_IFW_COMPONENT_GROUP_${_CPACK_IFWGRP_UNAME}_USER_INTERFACES)
  _cpack_ifw_resolve_file_list(CPACK_IFW_COMPONENT_GROUP_${_CPACK_IFWGRP_UNAME}_TRANSLATIONS)

  set(_CPACK_IFWGRP_STR "\n# Configuration for IFW component group \"${grpname}\"\n")

  foreach(_IFW_ARG_NAME ${_IFW_OPT})
  cpack_append_option_set_command(
    CPACK_IFW_COMPONENT_GROUP_${_CPACK_IFWGRP_UNAME}_${_IFW_ARG_NAME}
    _CPACK_IFWGRP_STR)
  endforeach()

  foreach(_IFW_ARG_NAME ${_IFW_ARGS})
  cpack_append_string_variable_set_command(
    CPACK_IFW_COMPONENT_GROUP_${_CPACK_IFWGRP_UNAME}_${_IFW_ARG_NAME}
    _CPACK_IFWGRP_STR)
  endforeach()

  foreach(_IFW_ARG_NAME ${_IFW_MULTI_ARGS})
  cpack_append_list_variable_set_command(
    CPACK_IFW_COMPONENT_GROUP_${_CPACK_IFWGRP_UNAME}_${_IFW_ARG_NAME}
    _CPACK_IFWGRP_STR)
  endforeach()

  if(CPack_CMake_INCLUDED)
    file(APPEND "${CPACK_OUTPUT_CONFIG_FILE}" "${_CPACK_IFWGRP_STR}")
  endif()
endmacro()

# Macro for adding repository
macro(cpack_ifw_add_repository reponame)

  string(TOUPPER ${reponame} _CPACK_IFWREPO_UNAME)

  set(_IFW_OPT DISABLED)
  set(_IFW_ARGS URL USERNAME PASSWORD DISPLAY_NAME)
  set(_IFW_MULTI_ARGS)
  cmake_parse_arguments(CPACK_IFW_REPOSITORY_${_CPACK_IFWREPO_UNAME} "${_IFW_OPT}" "${_IFW_ARGS}" "${_IFW_MULTI_ARGS}" ${ARGN})

  set(_CPACK_IFWREPO_STR "\n# Configuration for IFW repository \"${reponame}\"\n")

  foreach(_IFW_ARG_NAME ${_IFW_OPT})
  cpack_append_option_set_command(
    CPACK_IFW_REPOSITORY_${_CPACK_IFWREPO_UNAME}_${_IFW_ARG_NAME}
    _CPACK_IFWREPO_STR)
  endforeach()

  foreach(_IFW_ARG_NAME ${_IFW_ARGS})
  cpack_append_string_variable_set_command(
    CPACK_IFW_REPOSITORY_${_CPACK_IFWREPO_UNAME}_${_IFW_ARG_NAME}
    _CPACK_IFWREPO_STR)
  endforeach()

  foreach(_IFW_ARG_NAME ${_IFW_MULTI_ARGS})
  cpack_append_variable_set_command(
    CPACK_IFW_REPOSITORY_${_CPACK_IFWREPO_UNAME}_${_IFW_ARG_NAME}
    _CPACK_IFWREPO_STR)
  endforeach()

  list(APPEND CPACK_IFW_REPOSITORIES_ALL ${reponame})
  string(APPEND _CPACK_IFWREPO_STR "list(APPEND CPACK_IFW_REPOSITORIES_ALL ${reponame})\n")

  if(CPack_CMake_INCLUDED)
    file(APPEND "${CPACK_OUTPUT_CONFIG_FILE}" "${_CPACK_IFWREPO_STR}")
  endif()

endmacro()

# Macro for updating repository
macro(cpack_ifw_update_repository reponame)

  string(TOUPPER ${reponame} _CPACK_IFWREPO_UNAME)

  set(_IFW_OPT ADD REMOVE REPLACE DISABLED)
  set(_IFW_ARGS URL OLD_URL NEW_URL USERNAME PASSWORD DISPLAY_NAME)
  set(_IFW_MULTI_ARGS)
  cmake_parse_arguments(CPACK_IFW_REPOSITORY_${_CPACK_IFWREPO_UNAME} "${_IFW_OPT}" "${_IFW_ARGS}" "${_IFW_MULTI_ARGS}" ${ARGN})

  set(_CPACK_IFWREPO_STR "\n# Configuration for IFW repository \"${reponame}\" update\n")

  foreach(_IFW_ARG_NAME ${_IFW_OPT})
  cpack_append_option_set_command(
    CPACK_IFW_REPOSITORY_${_CPACK_IFWREPO_UNAME}_${_IFW_ARG_NAME}
    _CPACK_IFWREPO_STR)
  endforeach()

  foreach(_IFW_ARG_NAME ${_IFW_ARGS})
  cpack_append_string_variable_set_command(
    CPACK_IFW_REPOSITORY_${_CPACK_IFWREPO_UNAME}_${_IFW_ARG_NAME}
    _CPACK_IFWREPO_STR)
  endforeach()

  foreach(_IFW_ARG_NAME ${_IFW_MULTI_ARGS})
  cpack_append_variable_set_command(
    CPACK_IFW_REPOSITORY_${_CPACK_IFWREPO_UNAME}_${_IFW_ARG_NAME}
    _CPACK_IFWREPO_STR)
  endforeach()

  if(CPACK_IFW_REPOSITORY_${_CPACK_IFWREPO_UNAME}_ADD
    OR CPACK_IFW_REPOSITORY_${_CPACK_IFWREPO_UNAME}_REMOVE
    OR CPACK_IFW_REPOSITORY_${_CPACK_IFWREPO_UNAME}_REPLACE)
    list(APPEND CPACK_IFW_REPOSITORIES_ALL ${reponame})
    string(APPEND _CPACK_IFWREPO_STR "list(APPEND CPACK_IFW_REPOSITORIES_ALL ${reponame})\n")
  else()
    set(_CPACK_IFWREPO_STR)
  endif()

  if(CPack_CMake_INCLUDED AND _CPACK_IFWREPO_STR)
    file(APPEND "${CPACK_OUTPUT_CONFIG_FILE}" "${_CPACK_IFWREPO_STR}")
  endif()

endmacro()

# Macro for adding resources
macro(cpack_ifw_add_package_resources)
  set(_CPACK_IFW_PACKAGE_RESOURCES ${ARGV})
  _cpack_ifw_resolve_file_list(_CPACK_IFW_PACKAGE_RESOURCES)
  list(APPEND CPACK_IFW_PACKAGE_RESOURCES ${_CPACK_IFW_PACKAGE_RESOURCES})
  set(_CPACK_IFWQRC_STR "list(APPEND CPACK_IFW_PACKAGE_RESOURCES \"${_CPACK_IFW_PACKAGE_RESOURCES}\")\n")
  if(CPack_CMake_INCLUDED)
    file(APPEND "${CPACK_OUTPUT_CONFIG_FILE}" "${_CPACK_IFWQRC_STR}")
  endif()
endmacro()

# Resolve package control script
_cpack_ifw_resolve_script(CPACK_IFW_PACKAGE_CONTROL_SCRIPT)

endif() # NOT CPackIFW_CMake_INCLUDED
