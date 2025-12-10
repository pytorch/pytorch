# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CPackComponent
--------------

This module provides commands to configure components for binary installers
and source packages.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CPackComponent)

.. only:: html

  .. contents::

Introduction
^^^^^^^^^^^^

This module is also automatically included by :module:`CPack`.

Certain binary installers (especially the graphical installers) generated
by CPack allow users to select individual application *components* to install.
This module allows developers to configure the packaging of such components.

Contents is assigned to components by the ``COMPONENT``
argument of CMake's :command:`install` command.  Components can be
annotated with user-friendly names and descriptions, inter-component
dependencies, etc., and grouped in various ways to customize the
resulting installer, using the commands described below.

To specify different groupings for different CPack generators use
a CPACK_PROJECT_CONFIG_FILE.

Variables
^^^^^^^^^

The following variables influence the component-specific packaging:

.. variable:: CPACK_COMPONENTS_ALL

 The list of component to install.

 The default value of this variable is computed by CPack and contains all
 components defined by the project.  The user may set it to only include the
 specified components.

 Instead of specifying all the desired components, it is possible to obtain a
 list of all defined components and then remove the unwanted ones from the
 list. The :command:`get_cmake_property` command can be used to obtain the
 ``COMPONENTS`` property, then the :command:`list(REMOVE_ITEM)` command can be
 used to remove the unwanted ones. For example, to use all defined components
 except ``foo`` and ``bar``:

 .. code-block:: cmake

   get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)
   list(REMOVE_ITEM CPACK_COMPONENTS_ALL "foo" "bar")

.. variable:: CPACK_<GENNAME>_COMPONENT_INSTALL

 Enable/Disable component install for CPack generator <GENNAME>.

 Each CPack Generator (RPM, DEB, ARCHIVE, NSIS, DMG, etc...) has a legacy
 default behavior.  e.g.  RPM builds monolithic whereas NSIS builds
 component.  One can change the default behavior by setting this variable to
 0/1 or OFF/ON.

.. variable:: CPACK_COMPONENTS_GROUPING

 Specify how components are grouped for multi-package component-aware CPack
 generators.

 Some generators like RPM or ARCHIVE (TGZ, ZIP, ...) may generate
 several packages files when there are components, depending
 on the value of this variable:

 * ONE_PER_GROUP (default): create one package per component group
 * IGNORE : create one package per component (ignore the groups)
 * ALL_COMPONENTS_IN_ONE : create a single package with all requested
   components

.. variable:: CPACK_COMPONENT_<compName>_DISPLAY_NAME

 The name to be displayed for a component.

.. variable:: CPACK_COMPONENT_<compName>_DESCRIPTION

 The description of a component.

.. variable:: CPACK_COMPONENT_<compName>_GROUP

 The group of a component.

.. variable:: CPACK_COMPONENT_<compName>_DEPENDS

 The dependencies (list of components) on which this component depends.

.. variable:: CPACK_COMPONENT_<compName>_HIDDEN

 True if this component is hidden from the user.

.. variable:: CPACK_COMPONENT_<compName>_REQUIRED

 True if this component is required.

.. variable:: CPACK_COMPONENT_<compName>_DISABLED

 True if this component is not selected to be installed by default.

Commands
^^^^^^^^

Add component
"""""""""""""

.. command:: cpack_add_component

Describe an installation component.

.. code-block:: cmake

  cpack_add_component(compname
                      [DISPLAY_NAME name]
                      [DESCRIPTION description]
                      [HIDDEN | REQUIRED | DISABLED ]
                      [GROUP group]
                      [DEPENDS comp1 comp2 ... ]
                      [INSTALL_TYPES type1 type2 ... ]
                      [DOWNLOADED]
                      [ARCHIVE_FILE filename]
                      [PLIST filename])

``compname`` is the name of an installation component, as defined by the
``COMPONENT`` argument of one or more CMake :command:`install` commands.
With the ``cpack_add_component`` command one can set a name, a description,
and other attributes of an installation component.
One can also assign a component to a component group.

DISPLAY_NAME is the displayed name of the component, used in graphical
installers to display the component name.  This value can be any
string.

DESCRIPTION is an extended description of the component, used in
graphical installers to give the user additional information about the
component.  Descriptions can span multiple lines using ``\n`` as the
line separator.  Typically, these descriptions should be no more than
a few lines long.

HIDDEN indicates that this component will be hidden in the graphical
installer, so that the user cannot directly change whether it is
installed or not.

REQUIRED indicates that this component is required, and therefore will
always be installed.  It will be visible in the graphical installer,
but it cannot be unselected.  (Typically, required components are
shown grayed out).

DISABLED indicates that this component should be disabled (unselected)
by default.  The user is free to select this component for
installation, unless it is also HIDDEN.

DEPENDS lists the components on which this component depends.  If this
component is selected, then each of the components listed must also be
selected.  The dependency information is encoded within the installer
itself, so that users cannot install inconsistent sets of components.

GROUP names the component group of which this component is a part.  If
not provided, the component will be a standalone component, not part
of any component group.  Component groups are described with the
cpack_add_component_group command, detailed below.

INSTALL_TYPES lists the installation types of which this component is
a part.  When one of these installations types is selected, this
component will automatically be selected.  Installation types are
described with the cpack_add_install_type command, detailed below.

DOWNLOADED indicates that this component should be downloaded
on-the-fly by the installer, rather than packaged in with the
installer itself.  For more information, see the
cpack_configure_downloads command.

ARCHIVE_FILE provides a name for the archive file created by CPack to
be used for downloaded components.  If not supplied, CPack will create
a file with some name based on CPACK_PACKAGE_FILE_NAME and the name of
the component.  See cpack_configure_downloads for more information.

PLIST gives a filename that is passed to pkgbuild with the
``--component-plist`` argument when using the productbuild generator.

Add component group
"""""""""""""""""""

.. command:: cpack_add_component_group

Describes a group of related CPack installation components.

.. code-block:: cmake

  cpack_add_component_group(groupname
                           [DISPLAY_NAME name]
                           [DESCRIPTION description]
                           [PARENT_GROUP parent]
                           [EXPANDED]
                           [BOLD_TITLE])



The cpack_add_component_group describes a group of installation
components, which will be placed together within the listing of
options.  Typically, component groups allow the user to
select/deselect all of the components within a single group via a
single group-level option.  Use component groups to reduce the
complexity of installers with many options.  groupname is an arbitrary
name used to identify the group in the GROUP argument of the
cpack_add_component command, which is used to place a component in a
group.  The name of the group must not conflict with the name of any
component.

DISPLAY_NAME is the displayed name of the component group, used in
graphical installers to display the component group name.  This value
can be any string.

DESCRIPTION is an extended description of the component group, used in
graphical installers to give the user additional information about the
components within that group.  Descriptions can span multiple lines
using ``\n`` as the line separator.  Typically, these descriptions
should be no more than a few lines long.

PARENT_GROUP, if supplied, names the parent group of this group.
Parent groups are used to establish a hierarchy of groups, providing
an arbitrary hierarchy of groups.

EXPANDED indicates that, by default, the group should show up as
"expanded", so that the user immediately sees all of the components
within the group.  Otherwise, the group will initially show up as a
single entry.

BOLD_TITLE indicates that the group title should appear in bold, to
call the user's attention to the group.

Add installation type
"""""""""""""""""""""

.. command:: cpack_add_install_type

Add a new installation type containing
a set of predefined component selections to the graphical installer.

.. code-block:: cmake

  cpack_add_install_type(typename
                         [DISPLAY_NAME name])



The cpack_add_install_type command identifies a set of preselected
components that represents a common use case for an application.  For
example, a "Developer" install type might include an application along
with its header and library files, while an "End user" install type
might just include the application's executable.  Each component
identifies itself with one or more install types via the INSTALL_TYPES
argument to cpack_add_component.

DISPLAY_NAME is the displayed name of the install type, which will
typically show up in a drop-down box within a graphical installer.
This value can be any string.

Configure downloads
"""""""""""""""""""

.. command:: cpack_configure_downloads

Configure CPack to download
selected components on-the-fly as part of the installation process.

.. code-block:: cmake

  cpack_configure_downloads(site
                            [UPLOAD_DIRECTORY dirname]
                            [ALL]
                            [ADD_REMOVE|NO_ADD_REMOVE])



The cpack_configure_downloads command configures installation-time
downloads of selected components.  For each downloadable component,
CPack will create an archive containing the contents of that
component, which should be uploaded to the given site.  When the user
selects that component for installation, the installer will download
and extract the component in place.  This feature is useful for
creating small installers that only download the requested components,
saving bandwidth.  Additionally, the installers are small enough that
they will be installed as part of the normal installation process, and
the "Change" button in Windows Add/Remove Programs control panel will
allow one to add or remove parts of the application after the original
installation.  On Windows, the downloaded-components functionality
requires the ZipDLL plug-in for NSIS, available at:

::

  http://nsis.sourceforge.net/ZipDLL_plug-in

On macOS, installers that download components on-the-fly can only
be built and installed on system using macOS 10.5 or later.

The site argument is a URL where the archives for downloadable
components will reside, e.g.,
https://cmake.org/files/v3.25/ All of the archives
produced by CPack should be uploaded to that location.

UPLOAD_DIRECTORY is the local directory where CPack will create the
various archives for each of the components.  The contents of this
directory should be uploaded to a location accessible by the URL given
in the site argument.  If omitted, CPack will use the directory
CPackUploads inside the CMake binary directory to store the generated
archives.

The ALL flag indicates that all components be downloaded.  Otherwise,
only those components explicitly marked as DOWNLOADED or that have a
specified ARCHIVE_FILE will be downloaded.  Additionally, the ALL
option implies ADD_REMOVE (unless NO_ADD_REMOVE is specified).

ADD_REMOVE indicates that CPack should install a copy of the installer
that can be called from Windows' Add/Remove Programs dialog (via the
"Modify" button) to change the set of installed components.
NO_ADD_REMOVE turns off this behavior.  This option is ignored on Mac
OS X.
#]=======================================================================]

# Define var in order to avoid multiple inclusion
if(NOT CPackComponent_CMake_INCLUDED)
set(CPackComponent_CMake_INCLUDED 1)

# Function that appends a SET command for the given variable name (var)
# to the string named strvar, but only if the variable named "var"
# has been defined. The string will eventually be appended to a CPack
# configuration file.
function(cpack_append_variable_set_command var strvar)
  if (DEFINED ${var})
    string(APPEND ${strvar} "set(${var}")
    foreach(APPENDVAL ${${var}})
      string(APPEND ${strvar} " ${APPENDVAL}")
    endforeach()
    string(APPEND ${strvar} ")\n")
    set(${strvar} "${${strvar}}" PARENT_SCOPE)
  endif ()
endfunction()

# Function that appends a SET command for the given variable name (var)
# to the string named strvar, but only if the variable named "var"
# has been defined and is a string. The string will eventually be
# appended to a CPack configuration file.
function(cpack_append_string_variable_set_command var strvar)
  if (DEFINED ${var})
    list(LENGTH ${var} CPACK_APP_VALUE_LEN)
    if(${CPACK_APP_VALUE_LEN} EQUAL 1)
      string(APPEND ${strvar} "set(${var} \"${${var}}\")\n")
    endif()
    set(${strvar} "${${strvar}}" PARENT_SCOPE)
  endif ()
endfunction()

# Macro that appends a SET command for the given list variable name (var)
# to the macro named strvar, but only if the variable named "var"
# has been defined. It's like add variable, but wrap each item to quotes.
# The string will eventually be appended to a CPack configuration file.
macro(cpack_append_list_variable_set_command var strvar)
  if (DEFINED ${var})
    string(APPEND ${strvar} "set(${var}")
    foreach(_val IN LISTS ${var})
      string(APPEND ${strvar} "\n  \"${_val}\"")
    endforeach()
    string(APPEND ${strvar} ")\n")
  endif ()
endmacro()

# Macro that appends a SET command for the given variable name (var)
# to the macro named strvar, but only if the variable named "var"
# has been set to true. The string will eventually be
# appended to a CPack configuration file.
macro(cpack_append_option_set_command var strvar)
  if (${var})
    list(LENGTH ${var} CPACK_APP_VALUE_LEN)
    if(${CPACK_APP_VALUE_LEN} EQUAL 1)
      string(APPEND ${strvar} "set(${var} TRUE)\n")
    endif()
  endif ()
endmacro()

# Macro that adds a component to the CPack installer
macro(cpack_add_component compname)
  string(TOUPPER ${compname} _CPACK_ADDCOMP_UNAME)
  cmake_parse_arguments(CPACK_COMPONENT_${_CPACK_ADDCOMP_UNAME}
    "HIDDEN;REQUIRED;DISABLED;DOWNLOADED"
    "DISPLAY_NAME;DESCRIPTION;GROUP;ARCHIVE_FILE;PLIST"
    "DEPENDS;INSTALL_TYPES"
    ${ARGN}
    )

  if (CPACK_COMPONENT_${_CPACK_ADDCOMP_UNAME}_DOWNLOADED)
    set(_CPACK_ADDCOMP_STR "\n# Configuration for downloaded component \"${compname}\"\n")
  else ()
    set(_CPACK_ADDCOMP_STR "\n# Configuration for component \"${compname}\"\n")
  endif ()

  if(NOT CPACK_MONOLITHIC_INSTALL)
    # If the user didn't set CPACK_COMPONENTS_ALL explicitly, update the
    # value of CPACK_COMPONENTS_ALL in the configuration file. This will
    # take care of any components that have been added after the CPack
    # moduled was included.
    if(NOT CPACK_COMPONENTS_ALL_SET_BY_USER)
      get_cmake_property(_CPACK_ADDCOMP_COMPONENTS COMPONENTS)
      string(APPEND _CPACK_ADDCOMP_STR "\nSET(CPACK_COMPONENTS_ALL")
      foreach(COMP ${_CPACK_ADDCOMP_COMPONENTS})
       string(APPEND _CPACK_ADDCOMP_STR " ${COMP}")
      endforeach()
      string(APPEND _CPACK_ADDCOMP_STR ")\n")
    endif()
  endif()

  cpack_append_string_variable_set_command(
    CPACK_COMPONENT_${_CPACK_ADDCOMP_UNAME}_DISPLAY_NAME
    _CPACK_ADDCOMP_STR)
  cpack_append_string_variable_set_command(
    CPACK_COMPONENT_${_CPACK_ADDCOMP_UNAME}_DESCRIPTION
    _CPACK_ADDCOMP_STR)
  cpack_append_variable_set_command(
    CPACK_COMPONENT_${_CPACK_ADDCOMP_UNAME}_GROUP
    _CPACK_ADDCOMP_STR)
  cpack_append_variable_set_command(
    CPACK_COMPONENT_${_CPACK_ADDCOMP_UNAME}_DEPENDS
    _CPACK_ADDCOMP_STR)
  cpack_append_variable_set_command(
    CPACK_COMPONENT_${_CPACK_ADDCOMP_UNAME}_INSTALL_TYPES
    _CPACK_ADDCOMP_STR)
  cpack_append_string_variable_set_command(
    CPACK_COMPONENT_${_CPACK_ADDCOMP_UNAME}_ARCHIVE_FILE
    _CPACK_ADDCOMP_STR)
  cpack_append_option_set_command(
    CPACK_COMPONENT_${_CPACK_ADDCOMP_UNAME}_HIDDEN
    _CPACK_ADDCOMP_STR)
  cpack_append_option_set_command(
    CPACK_COMPONENT_${_CPACK_ADDCOMP_UNAME}_REQUIRED
    _CPACK_ADDCOMP_STR)
  cpack_append_option_set_command(
    CPACK_COMPONENT_${_CPACK_ADDCOMP_UNAME}_DISABLED
    _CPACK_ADDCOMP_STR)
  cpack_append_option_set_command(
    CPACK_COMPONENT_${_CPACK_ADDCOMP_UNAME}_DOWNLOADED
    _CPACK_ADDCOMP_STR)
  cpack_append_string_variable_set_command(
    CPACK_COMPONENT_${_CPACK_ADDCOMP_UNAME}_PLIST
    _CPACK_ADDCOMP_STR)
  # Backward compatibility issue.
  # Write to config iff the macros is used after CPack.cmake has been
  # included, other it's not necessary because the variables
  # will be encoded by cpack_encode_variables.
  if(CPack_CMake_INCLUDED)
    file(APPEND "${CPACK_OUTPUT_CONFIG_FILE}" "${_CPACK_ADDCOMP_STR}")
  endif()
endmacro()

# Macro that adds a component group to the CPack installer
macro(cpack_add_component_group grpname)
  string(TOUPPER ${grpname} _CPACK_ADDGRP_UNAME)
  cmake_parse_arguments(CPACK_COMPONENT_GROUP_${_CPACK_ADDGRP_UNAME}
    "EXPANDED;BOLD_TITLE"
    "DISPLAY_NAME;DESCRIPTION;PARENT_GROUP"
    ""
    ${ARGN}
    )

  set(_CPACK_ADDGRP_STR "\n# Configuration for component group \"${grpname}\"\n")
  cpack_append_string_variable_set_command(
    CPACK_COMPONENT_GROUP_${_CPACK_ADDGRP_UNAME}_DISPLAY_NAME
    _CPACK_ADDGRP_STR)
  cpack_append_string_variable_set_command(
    CPACK_COMPONENT_GROUP_${_CPACK_ADDGRP_UNAME}_DESCRIPTION
    _CPACK_ADDGRP_STR)
  cpack_append_string_variable_set_command(
    CPACK_COMPONENT_GROUP_${_CPACK_ADDGRP_UNAME}_PARENT_GROUP
    _CPACK_ADDGRP_STR)
  cpack_append_option_set_command(
    CPACK_COMPONENT_GROUP_${_CPACK_ADDGRP_UNAME}_EXPANDED
    _CPACK_ADDGRP_STR)
  cpack_append_option_set_command(
    CPACK_COMPONENT_GROUP_${_CPACK_ADDGRP_UNAME}_BOLD_TITLE
    _CPACK_ADDGRP_STR)
  # Backward compatibility issue.
  # Write to config iff the macros is used after CPack.cmake has been
  # included, other it's not necessary because the variables
  # will be encoded by cpack_encode_variables.
  if(CPack_CMake_INCLUDED)
    file(APPEND "${CPACK_OUTPUT_CONFIG_FILE}" "${_CPACK_ADDGRP_STR}")
  endif()
endmacro()

# Macro that adds an installation type to the CPack installer
macro(cpack_add_install_type insttype)
  string(TOUPPER ${insttype} _CPACK_INSTTYPE_UNAME)
  cmake_parse_arguments(CPACK_INSTALL_TYPE_${_CPACK_INSTTYPE_UNAME}
    ""
    "DISPLAY_NAME"
    ""
    ${ARGN}
    )

  set(_CPACK_INSTTYPE_STR
    "\n# Configuration for installation type \"${insttype}\"\n")
  string(APPEND _CPACK_INSTTYPE_STR
    "list(APPEND CPACK_ALL_INSTALL_TYPES ${insttype})\n")
  cpack_append_string_variable_set_command(
    CPACK_INSTALL_TYPE_${_CPACK_INSTTYPE_UNAME}_DISPLAY_NAME
    _CPACK_INSTTYPE_STR)
  # Backward compatibility issue.
  # Write to config iff the macros is used after CPack.cmake has been
  # included, other it's not necessary because the variables
  # will be encoded by cpack_encode_variables.
  if(CPack_CMake_INCLUDED)
    file(APPEND "${CPACK_OUTPUT_CONFIG_FILE}" "${_CPACK_INSTTYPE_STR}")
  endif()
endmacro()

macro(cpack_configure_downloads site)
  cmake_parse_arguments(CPACK_DOWNLOAD
    "ALL;ADD_REMOVE;NO_ADD_REMOVE"
    "UPLOAD_DIRECTORY"
    ""
    ${ARGN}
    )

  set(CPACK_CONFIG_DL_STR
    "\n# Downloaded components configuration\n")
  set(CPACK_UPLOAD_DIRECTORY ${CPACK_DOWNLOAD_UPLOAD_DIRECTORY})
  set(CPACK_DOWNLOAD_SITE ${site})
  cpack_append_string_variable_set_command(
    CPACK_DOWNLOAD_SITE
    CPACK_CONFIG_DL_STR)
  cpack_append_string_variable_set_command(
    CPACK_UPLOAD_DIRECTORY
    CPACK_CONFIG_DL_STR)
  cpack_append_option_set_command(
    CPACK_DOWNLOAD_ALL
    CPACK_CONFIG_DL_STR)
  if (${CPACK_DOWNLOAD_ALL} AND NOT ${CPACK_DOWNLOAD_NO_ADD_REMOVE})
    set(CPACK_DOWNLOAD_ADD_REMOVE ON)
  endif ()
  set(CPACK_ADD_REMOVE ${CPACK_DOWNLOAD_ADD_REMOVE})
  cpack_append_option_set_command(
    CPACK_ADD_REMOVE
    CPACK_CONFIG_DL_STR)
  # Backward compatibility issue.
  # Write to config iff the macros is used after CPack.cmake has been
  # included, other it's not necessary because the variables
  # will be encoded by cpack_encode_variables.
  if(CPack_CMake_INCLUDED)
    file(APPEND "${CPACK_OUTPUT_CONFIG_FILE}" "${CPACK_CONFIG_DL_STR}")
  endif()
endmacro()
endif()
