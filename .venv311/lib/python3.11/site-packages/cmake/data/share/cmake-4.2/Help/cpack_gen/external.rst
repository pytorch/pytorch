CPack External Generator
------------------------

.. versionadded:: 3.13

CPack provides many generators to create packages for a variety of platforms
and packaging systems. The intention is for CMake/CPack to be a complete
end-to-end solution for building and packaging a software project. However, it
may not always be possible to use CPack for the entire packaging process, due
to either technical limitations or policies that require the use of certain
tools. For this reason, CPack provides the "External" generator, which allows
external packaging software to take advantage of some of the functionality
provided by CPack, such as component installation and the dependency graph.

Integration with External Packaging Tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CPack External generator generates a ``.json`` file containing the
CPack internal metadata, which gives external software information
on how to package the software. External packaging software may itself
invoke CPack, consume the generated metadata,
install and package files as required.

Alternatively CPack can invoke an external packaging software
through an optional custom CMake script in
:variable:`CPACK_EXTERNAL_PACKAGE_SCRIPT` instead.

Staging of installation files may also optionally be
taken care of by the generator when enabled through the
:variable:`CPACK_EXTERNAL_ENABLE_STAGING` variable.

JSON Format
^^^^^^^^^^^

The JSON metadata file contains a list of CPack components and component groups,
the various options passed to :command:`cpack_add_component` and
:command:`cpack_add_component_group`, the dependencies between the components
and component groups, and various other options passed to CPack.

The JSON's root object will always provide two fields:
``formatVersionMajor`` and ``formatVersionMinor``, which are always integers
that describe the output format of the generator. Backwards-compatible changes
to the output format (for example, adding a new field that didn't exist before)
cause the minor version to be incremented, and backwards-incompatible changes
(for example, deleting a field or changing its meaning) cause the major version
to be incremented and the minor version reset to 0. The format version is
always of the format ``major.minor``. In other words, it always has exactly two
parts, separated by a period.

You can request one or more specific versions of the output format as described
below with :variable:`CPACK_EXTERNAL_REQUESTED_VERSIONS`. The output format will
have a major version that exactly matches the requested major version, and a
minor version that is greater than or equal to the requested minor version. If
no version is requested with :variable:`CPACK_EXTERNAL_REQUESTED_VERSIONS`, the
latest known major version is used by default. Currently, the only supported
format is 1.0, which is described below.

Version 1.0
***********

In addition to the standard format fields, format version 1.0 provides the
following fields in the root:

``components``
  The ``components`` field is an object with component names as the keys and
  objects describing the components as the values. The component objects have
  the following fields:

  ``name``
    The name of the component. This is always the same as the key in the
    ``components`` object.

  ``displayName``
    The value of the ``DISPLAY_NAME`` field passed to
    :command:`cpack_add_component`.

  ``description``
    The value of the ``DESCRIPTION`` field passed to
    :command:`cpack_add_component`.

  ``isHidden``
    True if ``HIDDEN`` was passed to :command:`cpack_add_component`, false if
    it was not.

  ``isRequired``
    True if ``REQUIRED`` was passed to :command:`cpack_add_component`, false if
    it was not.

  ``isDisabledByDefault``
    True if ``DISABLED`` was passed to :command:`cpack_add_component`, false if
    it was not.

  ``group``
    Only present if ``GROUP`` was passed to :command:`cpack_add_component`. If
    so, this field is a string value containing the component's group.

  ``dependencies``
    An array of components the component depends on. This contains the values
    in the ``DEPENDS`` argument passed to :command:`cpack_add_component`. If no
    ``DEPENDS`` argument was passed, this is an empty list.

  ``installationTypes``
    An array of installation types the component is part of. This contains the
    values in the ``INSTALL_TYPES`` argument passed to
    :command:`cpack_add_component`. If no ``INSTALL_TYPES`` argument was
    passed, this is an empty list.

  ``isDownloaded``
    True if ``DOWNLOADED`` was passed to :command:`cpack_add_component`, false
    if it was not.

  ``archiveFile``
    The name of the archive file passed with the ``ARCHIVE_FILE`` argument to
    :command:`cpack_add_component`. If no ``ARCHIVE_FILE`` argument was passed,
    this is an empty string.

``componentGroups``
  The ``componentGroups`` field is an object with component group names as the
  keys and objects describing the component groups as the values. The component
  group objects have the following fields:

  ``name``
    The name of the component group. This is always the same as the key in the
    ``componentGroups`` object.

  ``displayName``
    The value of the ``DISPLAY_NAME`` field passed to
    :command:`cpack_add_component_group`.

  ``description``
    The value of the ``DESCRIPTION`` field passed to
    :command:`cpack_add_component_group`.

  ``parentGroup``
    Only present if ``PARENT_GROUP`` was passed to
    :command:`cpack_add_component_group`. If so, this field is a string value
    containing the component group's parent group.

  ``isExpandedByDefault``
    True if ``EXPANDED`` was passed to :command:`cpack_add_component_group`,
    false if it was not.

  ``isBold``
    True if ``BOLD_TITLE`` was passed to :command:`cpack_add_component_group`,
    false if it was not.

  ``components``
    An array of names of components that are direct members of the group
    (components that have this group as their ``GROUP``). Components of
    subgroups are not included.

  ``subgroups``
    An array of names of component groups that are subgroups of the group
    (groups that have this group as their ``PARENT_GROUP``).

``installationTypes``
  The ``installationTypes`` field is an object with installation type names as
  the keys and objects describing the installation types as the values. The
  installation type objects have the following fields:

  ``name``
    The name of the installation type. This is always the same as the key in
    the ``installationTypes`` object.

  ``displayName``
    The value of the ``DISPLAY_NAME`` field passed to
    :command:`cpack_add_install_type`.

  ``index``
    The integer index of the installation type in the list.

``projects``
  The ``projects`` field is an array of objects describing CMake projects which
  comprise the CPack project. The values in this field are derived from
  :variable:`CPACK_INSTALL_CMAKE_PROJECTS`. In most cases, this will be only a
  single project. The project objects have the following fields:

  ``projectName``
    The project name passed to :variable:`CPACK_INSTALL_CMAKE_PROJECTS`.

  ``component``
    The name of the component or component set which comprises the project.

  ``directory``
    The build directory of the CMake project. This is the directory which
    contains the ``cmake_install.cmake`` script.

  ``subDirectory``
    The subdirectory to install the project into inside the CPack package.

``packageName``
  The package name given in :variable:`CPACK_PACKAGE_NAME`. Only present if
  this option is set.

``packageVersion``
  The package version given in :variable:`CPACK_PACKAGE_VERSION`. Only present
  if this option is set.

``packageDescriptionFile``
  The package description file given in
  :variable:`CPACK_PACKAGE_DESCRIPTION_FILE`. Only present if this option is
  set.

``packageDescriptionSummary``
  The package description summary given in
  :variable:`CPACK_PACKAGE_DESCRIPTION_SUMMARY`. Only present if this option is
  set.

``buildConfig``
  The build configuration given to CPack with the :option:`cpack -C` option.
  Only present if this option is set.

``defaultDirectoryPermissions``
  The default directory permissions given in
  :variable:`CPACK_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS`. Only present if this
  option is set.

``setDestdir``
  True if :variable:`CPACK_SET_DESTDIR` is true, false if it is not.

``packagingInstallPrefix``
  The install prefix given in :variable:`CPACK_PACKAGING_INSTALL_PREFIX`. Only
  present if :variable:`CPACK_SET_DESTDIR` is true.

``stripFiles``
  True if :variable:`CPACK_STRIP_FILES` is true, false if it is not.

``warnOnAbsoluteInstallDestination``
  True if :variable:`CPACK_WARN_ON_ABSOLUTE_INSTALL_DESTINATION` is true, false
  if it is not.

``errorOnAbsoluteInstallDestination``
  True if :variable:`CPACK_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION` is true,
  false if it is not.

Variables specific to CPack External generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. variable:: CPACK_EXTERNAL_REQUESTED_VERSIONS

  This variable is used to request a specific version of the CPack External
  generator. It is a list of ``major.minor`` values, separated by semicolons.

  If this variable is set to a non-empty value, the CPack External generator
  will iterate through each item in the list to search for a version that it
  knows how to generate. Requested versions should be listed in order of
  descending preference by the client software, as the first matching version
  in the list will be generated.

  The generator knows how to generate the version if it has a versioned
  generator whose major version exactly matches the requested major version,
  and whose minor version is greater than or equal to the requested minor
  version. For example, if ``CPACK_EXTERNAL_REQUESTED_VERSIONS`` contains 1.0, and
  the CPack External generator knows how to generate 1.1, it will generate 1.1.
  If the generator doesn't know how to generate a version in the list, it skips
  the version and looks at the next one. If it doesn't know how to generate any
  of the requested versions, an error is thrown.

  If this variable is not set, or is empty, the CPack External generator will
  generate the highest major and minor version that it knows how to generate.

  If an invalid version is encountered in ``CPACK_EXTERNAL_REQUESTED_VERSIONS`` (one
  that doesn't match ``major.minor``, where ``major`` and ``minor`` are
  integers), it is ignored.

.. variable:: CPACK_EXTERNAL_ENABLE_STAGING

  This variable can be set to true to enable optional installation
  into a temporary staging area which can then be picked up
  and packaged by an external packaging tool.
  The top level directory used by CPack for the current packaging
  task is contained in ``CPACK_TOPLEVEL_DIRECTORY``.
  It is automatically cleaned up on each run before packaging is initiated
  and can be used for custom temporary files required by
  the external packaging tool.
  It also contains the staging area ``CPACK_TEMPORARY_DIRECTORY``
  into which CPack performs the installation when staging is enabled.

.. variable:: CPACK_EXTERNAL_PACKAGE_SCRIPT

  This variable can optionally specify the full path to
  a CMake script file to be run as part of the CPack invocation.
  It is invoked after (optional) staging took place and may
  run an external packaging tool. The script has access to
  the variables defined by the CPack config file.

.. variable:: CPACK_EXTERNAL_BUILT_PACKAGES

  .. versionadded:: 3.19

  The ``CPACK_EXTERNAL_PACKAGE_SCRIPT`` script may set this list variable to the
  full paths of generated package files.  CPack will copy these files from the
  staging directory back to the top build directory and possibly produce
  checksum files if the :variable:`CPACK_PACKAGE_CHECKSUM` is set.
