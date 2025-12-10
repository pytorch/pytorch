install
-------

.. only:: html

   .. contents::

Specify rules to run at install time.

Synopsis
^^^^^^^^

.. parsed-literal::

  install(`TARGETS`_ <target>... [...])
  install(`IMPORTED_RUNTIME_ARTIFACTS`_ <target>... [...])
  install({`FILES`_ | `PROGRAMS`_} <file>... [...])
  install(`DIRECTORY`_ <dir>... [...])
  install(`SCRIPT`_ <file> [...])
  install(`CODE`_ <code> [...])
  install(`EXPORT`_ <export-name> [...])
  install(`PACKAGE_INFO`_ <package-name> [...])
  install(`RUNTIME_DEPENDENCY_SET`_ <set-name> [...])

Introduction
^^^^^^^^^^^^

This command generates installation rules for a project.  Install rules
specified by calls to the ``install()`` command within a source directory
are executed in order during installation.

.. versionchanged:: 3.14
  Install rules in subdirectories
  added by calls to the :command:`add_subdirectory` command are interleaved
  with those in the parent directory to run in the order declared (see
  policy :policy:`CMP0082`).

.. versionchanged:: 3.22
  The environment variable :envvar:`CMAKE_INSTALL_MODE` can override the
  default copying behavior of ``install()``.

.. versionchanged:: 3.31
  Projects can enable :prop_gbl:`INSTALL_PARALLEL` to enable a parallel
  installation. When using the parallel install, subdirectories added by calls
  to the :command:`add_subdirectory` command are installed independently
  and the order that install rules added in different subdirectories will run is
  not guaranteed.

Common Options
""""""""""""""

There are multiple signatures for this command.  Some of them define
installation options for files and targets.  Options common to
multiple signatures are covered here but they are valid only for
signatures that specify them.  The common options are:

``DESTINATION <dir>``
  Specify the directory on disk to which a file will be installed.
  ``<dir>`` should be a relative path.  An absolute path is allowed,
  but not recommended.

  When a relative path is given, it is interpreted relative to the value
  of the :variable:`CMAKE_INSTALL_PREFIX` variable.
  The prefix can be relocated at install time using the ``DESTDIR``
  mechanism explained in the :variable:`CMAKE_INSTALL_PREFIX` variable
  documentation.

  As absolute paths do not work with the ``cmake --install`` command's
  :option:`--prefix <cmake--install --prefix>` option, or with the
  :manual:`cpack <cpack(1)>` installer generators, it is strongly recommended
  to use relative paths throughout for best support by package maintainers.
  In particular, there is no need to make paths absolute by prepending
  :variable:`CMAKE_INSTALL_PREFIX`; this prefix is used by default if
  the DESTINATION is a relative path.

  If an absolute path (with a leading slash or drive letter) is given
  it is used verbatim.

  .. versionchanged:: 3.31
    ``<dir>`` will be normalized according to the same
    :ref:`normalization rules <Normalization>` as the
    :command:`cmake_path` command.

``PERMISSIONS <permission>...``
  Specify permissions for installed files.  Valid permissions are
  ``OWNER_READ``, ``OWNER_WRITE``, ``OWNER_EXECUTE``, ``GROUP_READ``,
  ``GROUP_WRITE``, ``GROUP_EXECUTE``, ``WORLD_READ``, ``WORLD_WRITE``,
  ``WORLD_EXECUTE``, ``SETUID``, and ``SETGID``.  Permissions that do
  not make sense on certain platforms are ignored on those platforms.

  If this option is used multiple times in a single call, its list
  of permissions accumulates.  If an :command:`install(TARGETS)` call
  uses `\<artifact-kind\>`_ arguments, a separate list of permissions
  is accumulated for each kind of artifact.

``CONFIGURATIONS <config>...``
  Specify a list of build configurations for which the install rule
  applies (Debug, Release, etc.).

  If this option is used multiple times in a single call, its list
  of configurations accumulates.  If an :command:`install(TARGETS)`
  call uses `\<artifact-kind\>`_ arguments, a separate list of
  configurations is accumulated for each kind of artifact.

``COMPONENT <component>``
  Specify an installation component name with which the install rule
  is associated, such as ``Runtime`` or ``Development``.  During
  component-specific installation only install rules associated with
  the given component name will be executed.  During a full installation
  all components are installed unless marked with ``EXCLUDE_FROM_ALL``.
  If ``COMPONENT`` is not provided a default component "Unspecified" is
  created.  The default component name may be controlled with the
  :variable:`CMAKE_INSTALL_DEFAULT_COMPONENT_NAME` variable.

  Installation components can be then used by the ``cmake --install`` command's
  :option:`--component <cmake--install --component>` option and the
  :module:`CPackComponent` module.  Global target ``list_install_components``
  lists all available components.

``EXCLUDE_FROM_ALL``
  .. versionadded:: 3.6

  Specify that the file is excluded from a full installation and only
  installed as part of a component-specific installation

``OPTIONAL``
  Specify that it is not an error if the file to be installed does
  not exist.

.. versionadded:: 3.1
  Command signatures that install files may print messages during
  installation.  Use the :variable:`CMAKE_INSTALL_MESSAGE` variable
  to control which messages are printed.

.. versionadded:: 3.11
  Many of the ``install()`` variants implicitly create the directories
  containing the installed files. If
  :variable:`CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS` is set, these
  directories will be created with the permissions specified. Otherwise,
  they will be created according to the uname rules on Unix-like platforms.
  Windows platforms are unaffected.

Signatures
^^^^^^^^^^

.. signature::
  install(TARGETS <target>... [...])

  Install target :ref:`Output Artifacts` and associated files:

  .. code-block:: cmake

    install(TARGETS <target>... [EXPORT <export-name>]
            [RUNTIME_DEPENDENCIES <arg>...|RUNTIME_DEPENDENCY_SET <set-name>]
            [<artifact-option>...]
            [<artifact-kind> <artifact-option>...]...
            [INCLUDES DESTINATION [<dir> ...]]
            )

  where ``<artifact-option>...`` group may contain:

  .. code-block:: cmake

    [DESTINATION <dir>]
    [PERMISSIONS <permission>...]
    [CONFIGURATIONS <config>...]
    [COMPONENT <component>]
    [NAMELINK_COMPONENT <component>]
    [OPTIONAL] [EXCLUDE_FROM_ALL]
    [NAMELINK_ONLY|NAMELINK_SKIP]

  The first ``<artifact-option>...`` group applies to target
  :ref:`Output Artifacts` that do not have a dedicated group specified
  later in the same call.

  .. _`<artifact-kind>`:

  Each ``<artifact-kind> <artifact-option>...`` group applies to
  :ref:`Output Artifacts` of the specified artifact kind:

  ``ARCHIVE``
    Target artifacts of this kind include:

    * *Static libraries*
      (except on macOS when marked as ``FRAMEWORK``, see below);
    * *DLL import libraries*
      (on all Windows-based systems including Cygwin; they have extension
      ``.lib``, in contrast to the ``.dll`` libraries that go to ``RUNTIME``);
    * On AIX, the *linker import file* created for executables with
      :prop_tgt:`ENABLE_EXPORTS` enabled.
    * On macOS, the *linker import file* created for shared libraries with
      :prop_tgt:`ENABLE_EXPORTS` enabled (except when marked as ``FRAMEWORK``,
      see below).

  ``LIBRARY``
    Target artifacts of this kind include:

    * *Shared libraries*, except

      - DLLs (these go to ``RUNTIME``, see below),
      - on macOS when marked as ``FRAMEWORK`` (see below).

  ``RUNTIME``
    Target artifacts of this kind include:

    * *Executables*
      (except on macOS when marked as ``MACOSX_BUNDLE``, see ``BUNDLE`` below);
    * DLLs (on all Windows-based systems including Cygwin; note that the
      accompanying import libraries are of kind ``ARCHIVE``).

  ``OBJECTS``
    .. versionadded:: 3.9

    Object files associated with *object libraries*.

  ``FRAMEWORK``
    Both static and shared libraries marked with the ``FRAMEWORK``
    property are treated as ``FRAMEWORK`` targets on macOS.

  ``BUNDLE``
    Executables marked with the :prop_tgt:`MACOSX_BUNDLE` property are treated as
    ``BUNDLE`` targets on macOS.

  ``PUBLIC_HEADER``
    Any :prop_tgt:`PUBLIC_HEADER` files associated with a library are installed in
    the destination specified by the ``PUBLIC_HEADER`` argument on non-Apple
    platforms. Rules defined by this argument are ignored for :prop_tgt:`FRAMEWORK`
    libraries on Apple platforms because the associated files are installed
    into the appropriate locations inside the framework folder. See
    :prop_tgt:`PUBLIC_HEADER` for details.

  ``PRIVATE_HEADER``
    Similar to ``PUBLIC_HEADER``, but for ``PRIVATE_HEADER`` files. See
    :prop_tgt:`PRIVATE_HEADER` for details.

  ``RESOURCE``
    Similar to ``PUBLIC_HEADER`` and ``PRIVATE_HEADER``, but for
    ``RESOURCE`` files. See :prop_tgt:`RESOURCE` for details.

  ``FILE_SET <set-name>``
    .. versionadded:: 3.23

    File sets are defined by the :command:`target_sources(FILE_SET)` command.
    If the file set ``<set-name>`` exists and is ``PUBLIC`` or ``INTERFACE``,
    any files in the set are installed under the destination (see below).
    The directory structure relative to the file set's base directories is
    preserved. For example, a file added to the file set as
    ``/blah/include/myproj/here.h`` with a base directory ``/blah/include``
    would be installed to ``myproj/here.h`` below the destination.

  ``CXX_MODULES_BMI``
    .. versionadded:: 3.28

    Any module files from C++ modules from ``PUBLIC`` sources in a file set of
    type ``CXX_MODULES`` will be installed to the given ``DESTINATION``. All
    modules are placed directly in the destination as no directory structure is
    derived from the names of the modules. An empty ``DESTINATION`` may be used
    to suppress installing these files (for use in generic code).

  For regular executables, static libraries and shared libraries, the
  ``DESTINATION`` argument is not required.  For these target types, when
  ``DESTINATION`` is omitted, a default destination will be taken from the
  appropriate variable from :module:`GNUInstallDirs`, or set to a built-in
  default value if that variable is not defined.  The same is true for file
  sets, and the public and private headers associated with the installed
  targets through the :prop_tgt:`PUBLIC_HEADER` and :prop_tgt:`PRIVATE_HEADER`
  target properties. A destination must always be provided for module libraries,
  Apple bundles and frameworks.  A destination can be omitted for interface and
  object libraries, but they are handled differently (see the discussion of this
  topic toward the end of this section).

  For shared libraries on DLL platforms, if neither ``RUNTIME`` nor ``ARCHIVE``
  destinations are specified, both the ``RUNTIME`` and ``ARCHIVE`` components are
  installed to their default destinations. If either a ``RUNTIME`` or ``ARCHIVE``
  destination is specified, the component is installed to that destination, and
  the other component is not installed. If both ``RUNTIME`` and ``ARCHIVE``
  destinations are specified, then both components are installed to their
  respective destinations.

  The following table shows the target types with their associated variables and
  built-in defaults that apply when no destination is given:

  =============================== =============================== ======================
     Target Type                      GNUInstallDirs Variable        Built-In Default
  =============================== =============================== ======================
  ``RUNTIME``                     ``${CMAKE_INSTALL_BINDIR}``     ``bin``
  ``LIBRARY``                     ``${CMAKE_INSTALL_LIBDIR}``     ``lib``
  ``ARCHIVE``                     ``${CMAKE_INSTALL_LIBDIR}``     ``lib``
  ``PRIVATE_HEADER``              ``${CMAKE_INSTALL_INCLUDEDIR}`` ``include``
  ``PUBLIC_HEADER``               ``${CMAKE_INSTALL_INCLUDEDIR}`` ``include``
  ``FILE_SET`` (type ``HEADERS``) ``${CMAKE_INSTALL_INCLUDEDIR}`` ``include``
  =============================== =============================== ======================

  Projects wishing to follow the common practice of installing headers into a
  project-specific subdirectory may prefer using file sets with appropriate
  paths and base directories. Otherwise, they must provide a ``DESTINATION``
  instead of being able to rely on the above (see next example below).

  To make packages compliant with distribution filesystem layout policies, if
  projects must specify a ``DESTINATION``, it is strongly recommended that they use
  a path that begins with the appropriate relative :module:`GNUInstallDirs` variable.
  This allows package maintainers to control the install destination by setting
  the appropriate cache variables.  The following example shows a static library
  being installed to the default destination provided by
  :module:`GNUInstallDirs`, but with its headers installed to a project-specific
  subdirectory without using file sets:

  .. code-block:: cmake

    add_library(mylib STATIC ...)
    set_target_properties(mylib PROPERTIES PUBLIC_HEADER mylib.h)
    include(GNUInstallDirs)
    install(TARGETS mylib
            PUBLIC_HEADER
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/myproj
    )

  In addition to the `common options`_ listed above, each target can accept
  the following additional arguments:

  ``NAMELINK_COMPONENT``
    .. versionadded:: 3.12

    On some platforms a versioned shared library has a symbolic link such
    as::

      lib<name>.so -> lib<name>.so.1

    where ``lib<name>.so.1`` is the soname of the library and ``lib<name>.so``
    is a "namelink" allowing linkers to find the library when given
    ``-l<name>``. The ``NAMELINK_COMPONENT`` option is similar to the
    ``COMPONENT`` option, but it changes the installation component of a shared
    library namelink if one is generated. If not specified, this defaults to the
    value of ``COMPONENT``. It is an error to use this parameter outside of a
    ``LIBRARY`` block.

    .. versionchanged:: 3.27
      This parameter is also usable for an ``ARCHIVE`` block to manage
      the linker import file created, on macOS, for shared libraries with
      :prop_tgt:`ENABLE_EXPORTS` enabled.

    See the `Example: Install Targets with Per-Artifact Components`_
    for an example using ``NAMELINK_COMPONENT``.

    This option is typically used for package managers that have separate
    runtime and development packages. For example, on Debian systems, the
    library is expected to be in the runtime package, and the headers and
    namelink are expected to be in the development package.

    See the :prop_tgt:`VERSION` and :prop_tgt:`SOVERSION` target properties for
    details on creating versioned shared libraries.

  ``NAMELINK_ONLY``
    This option causes the installation of only the namelink when a library
    target is installed. On platforms where versioned shared libraries do not
    have namelinks or when a library is not versioned, the ``NAMELINK_ONLY``
    option installs nothing. It is an error to use this parameter outside of a
    ``LIBRARY`` block.

    .. versionchanged:: 3.27
      This parameter is also usable for an ``ARCHIVE`` block to manage
      the linker import file created, on macOS, for shared libraries with
      :prop_tgt:`ENABLE_EXPORTS` enabled.

    When ``NAMELINK_ONLY`` is given, either ``NAMELINK_COMPONENT`` or
    ``COMPONENT`` may be used to specify the installation component of the
    namelink, but ``COMPONENT`` should generally be preferred.

  ``NAMELINK_SKIP``
    Similar to ``NAMELINK_ONLY``, but it has the opposite effect: it causes the
    installation of library files other than the namelink when a library target
    is installed. When neither ``NAMELINK_ONLY`` or ``NAMELINK_SKIP`` are given,
    both portions are installed. On platforms where versioned shared libraries
    do not have symlinks or when a library is not versioned, ``NAMELINK_SKIP``
    installs the library. It is an error to use this parameter outside of a
    ``LIBRARY`` block.

    .. versionchanged:: 3.27
      This parameter is also usable for an ``ARCHIVE`` block to manage
      the linker import file created, on macOS, for shared libraries with
      :prop_tgt:`ENABLE_EXPORTS` enabled.

    If ``NAMELINK_SKIP`` is specified, ``NAMELINK_COMPONENT`` has no effect. It
    is not recommended to use ``NAMELINK_SKIP`` in conjunction with
    ``NAMELINK_COMPONENT``.

  The :command:`install(TARGETS)` command can also accept the following
  options at the top level:

  ``EXPORT``
    This option associates the installed target files with an export called
    ``<export-name>``.  It must appear before any target options.
    To actually install the export file itself, call
    :command:`install(EXPORT)`, documented below.
    See documentation of the :prop_tgt:`EXPORT_NAME` target property to change
    the name of the exported target.

    If ``EXPORT`` is used and the targets include ``PUBLIC`` or ``INTERFACE``
    file sets, all of them must be specified with ``FILE_SET`` arguments. All
    ``PUBLIC`` or ``INTERFACE`` file sets associated with a target are included
    in the export.

  ``INCLUDES DESTINATION``
    This option specifies a list of directories which will be added to the
    :prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` target property of the
    ``<targets>`` when exported by the :command:`install(EXPORT)` command.
    If a relative path is specified, it is treated as relative to the
    :genex:`$<INSTALL_PREFIX>`.

    Unlike other ``DESTINATION`` arguments for the various ``install()``
    subcommands, paths given after ``INCLUDES DESTINATION`` are used as
    given.  They are not normalized, nor assumed to be normalized, although
    it is recommended that they are given in normalized form (see
    :ref:`Normalization`).

  ``RUNTIME_DEPENDENCY_SET <set-name>``
    .. versionadded:: 3.21

    This option causes all runtime dependencies of installed executable, shared
    library, and module targets to be added to the specified runtime dependency
    set. This set can then be installed with an
    :command:`install(RUNTIME_DEPENDENCY_SET)` command.

    This keyword and the ``RUNTIME_DEPENDENCIES`` keyword are mutually
    exclusive.

  ``RUNTIME_DEPENDENCIES <arg>...``
    .. versionadded:: 3.21

    This option causes all runtime dependencies of installed executable, shared
    library, and module targets to be installed along with the targets
    themselves. The ``RUNTIME``, ``LIBRARY``, ``FRAMEWORK``, and generic
    arguments are used to determine the properties (``DESTINATION``,
    ``COMPONENT``, etc.) of the installation of these dependencies.

    ``RUNTIME_DEPENDENCIES`` is semantically equivalent to the following pair
    of calls:

    .. code-block:: cmake

      install(TARGETS ... RUNTIME_DEPENDENCY_SET <set-name>)
      install(RUNTIME_DEPENDENCY_SET <set-name> <arg>...)

    where ``<set-name>`` will be a randomly generated set name.
    ``<arg>...`` may include any of the following keywords supported by
    the :command:`install(RUNTIME_DEPENDENCY_SET)` command:

    * ``DIRECTORIES``
    * ``PRE_INCLUDE_REGEXES``
    * ``PRE_EXCLUDE_REGEXES``
    * ``POST_INCLUDE_REGEXES``
    * ``POST_EXCLUDE_REGEXES``
    * ``POST_INCLUDE_FILES``
    * ``POST_EXCLUDE_FILES``

    The ``RUNTIME_DEPENDENCIES`` and ``RUNTIME_DEPENDENCY_SET`` keywords are
    mutually exclusive.

  :ref:`Interface Libraries` may be listed among the targets to install.
  They install no artifacts but will be included in an associated ``EXPORT``.
  If :ref:`Object Libraries` are listed but given no destination for their
  object files, they will be exported as :ref:`Interface Libraries`.
  This is sufficient to satisfy transitive usage requirements of other
  targets that link to the object libraries in their implementation.

  Installing a target with the :prop_tgt:`EXCLUDE_FROM_ALL` target property
  set to ``TRUE`` has undefined behavior.

  .. versionadded:: 3.3
    An install destination given as a ``DESTINATION`` argument may
    use "generator expressions" with the syntax ``$<...>``.  See the
    :manual:`cmake-generator-expressions(7)` manual for available expressions.

  .. versionadded:: 3.13
    :command:`install(TARGETS)` can install targets that were created in
    other directories.  When using such cross-directory install rules, running
    ``make install`` (or similar) from a subdirectory will not guarantee that
    targets from other directories are up-to-date.  You can use
    :command:`target_link_libraries` or :command:`add_dependencies`
    to ensure that such out-of-directory targets are built before the
    subdirectory-specific install rules are run.

.. signature::
  install(IMPORTED_RUNTIME_ARTIFACTS <target>... [...])

  .. versionadded:: 3.21

  Install runtime artifacts of imported targets:

  .. code-block:: cmake

    install(IMPORTED_RUNTIME_ARTIFACTS <target>...
            [RUNTIME_DEPENDENCY_SET <set-name>]
            [[LIBRARY|RUNTIME|FRAMEWORK|BUNDLE]
             [DESTINATION <dir>]
             [PERMISSIONS <permission>...]
             [CONFIGURATIONS <config>...]
             [COMPONENT <component>]
             [OPTIONAL] [EXCLUDE_FROM_ALL]
            ] [...]
            )

  The ``IMPORTED_RUNTIME_ARTIFACTS`` form specifies rules for installing the
  runtime artifacts of imported targets. Projects may do this if they want to
  bundle outside executables or modules inside their installation. The
  ``LIBRARY``, ``RUNTIME``, ``FRAMEWORK``, and ``BUNDLE`` arguments have the
  same semantics that they do in the `TARGETS`_ mode. Only the runtime artifacts
  of imported targets are installed (except in the case of :prop_tgt:`FRAMEWORK`
  libraries, :prop_tgt:`MACOSX_BUNDLE` executables, and :prop_tgt:`BUNDLE`
  CFBundles.) For example, headers and import libraries associated with DLLs are
  not installed. In the case of :prop_tgt:`FRAMEWORK` libraries,
  :prop_tgt:`MACOSX_BUNDLE` executables, and :prop_tgt:`BUNDLE` CFBundles, the
  entire directory is installed.

  The ``RUNTIME_DEPENDENCY_SET`` option causes the runtime artifacts of the
  imported executable, shared library, and module library ``targets`` to be
  added to the ``<set-name>`` runtime dependency set. This set can then be
  installed with an :command:`install(RUNTIME_DEPENDENCY_SET)` command.

.. signature::
  install(FILES <file>... [...])
  install(PROGRAMS <program>... [...])

  .. note::

    If installing header files, consider using file sets defined by
    :command:`target_sources(FILE_SET)` instead. File sets associate
    headers with a target and they install as part of the target.

  Install files or programs:

  .. code-block:: cmake

    install(<FILES|PROGRAMS> <file>...
            TYPE <type> | DESTINATION <dir>
            [PERMISSIONS <permission>...]
            [CONFIGURATIONS <config>...]
            [COMPONENT <component>]
            [RENAME <name>] [OPTIONAL] [EXCLUDE_FROM_ALL])

  The ``FILES`` form specifies rules for installing files for a project.
  File names given as relative paths are interpreted with respect to the
  current source directory.  Files installed by this form are by default
  given permissions ``OWNER_WRITE``, ``OWNER_READ``, ``GROUP_READ``, and
  ``WORLD_READ`` if no ``PERMISSIONS`` argument is given.

  The ``PROGRAMS`` form is identical to the ``FILES`` form except that the
  default permissions for the installed file also include ``OWNER_EXECUTE``,
  ``GROUP_EXECUTE``, and ``WORLD_EXECUTE``.  This form is intended to install
  programs that are not targets, such as shell scripts.  Use the ``TARGETS``
  form to install targets built within the project.

  The list of ``files...`` given to ``FILES`` or ``PROGRAMS`` may use
  "generator expressions" with the syntax ``$<...>``.  See the
  :manual:`cmake-generator-expressions(7)` manual for available expressions.
  However, if any item begins in a generator expression it must evaluate
  to a full path.

  The optional ``RENAME <name>`` argument is used to specify a name for the
  installed file that is different from the original file name.  Renaming
  is allowed only when a single file is installed by the command.

  Either a ``TYPE`` or a ``DESTINATION`` must be provided, but not both.
  A ``TYPE`` argument specifies the generic file type of the files being
  installed.  A destination will then be set automatically by taking the
  corresponding variable from :module:`GNUInstallDirs`, or by using a
  built-in default if that variable is not defined.  See the table below for
  the supported file types and their corresponding variables and built-in
  defaults.  Projects can provide a ``DESTINATION`` argument instead of a
  file type if they wish to explicitly define the install destination.

  ======================= ================================== =========================
     ``TYPE`` Argument         GNUInstallDirs Variable           Built-In Default
  ======================= ================================== =========================
  ``BIN``                 ``${CMAKE_INSTALL_BINDIR}``        ``bin``
  ``SBIN``                ``${CMAKE_INSTALL_SBINDIR}``       ``sbin``
  ``LIB``                 ``${CMAKE_INSTALL_LIBDIR}``        ``lib``
  ``INCLUDE``             ``${CMAKE_INSTALL_INCLUDEDIR}``    ``include``
  ``SYSCONF``             ``${CMAKE_INSTALL_SYSCONFDIR}``    ``etc``
  ``SHAREDSTATE``         ``${CMAKE_INSTALL_SHARESTATEDIR}`` ``com``
  ``LOCALSTATE``          ``${CMAKE_INSTALL_LOCALSTATEDIR}`` ``var``
  ``RUNSTATE``            ``${CMAKE_INSTALL_RUNSTATEDIR}``   ``<LOCALSTATE dir>/run``
  ``DATA``                ``${CMAKE_INSTALL_DATADIR}``       ``<DATAROOT dir>``
  ``INFO``                ``${CMAKE_INSTALL_INFODIR}``       ``<DATAROOT dir>/info``
  ``LOCALE``              ``${CMAKE_INSTALL_LOCALEDIR}``     ``<DATAROOT dir>/locale``
  ``MAN``                 ``${CMAKE_INSTALL_MANDIR}``        ``<DATAROOT dir>/man``
  ``DOC``                 ``${CMAKE_INSTALL_DOCDIR}``        ``<DATAROOT dir>/doc``
  ``LIBEXEC``             ``${CMAKE_INSTALL_LIBEXECDIR}``    ``libexec``
  ======================= ================================== =========================

  Projects wishing to follow the common practice of installing headers into a
  project-specific subdirectory will need to provide a destination rather than
  rely on the above. Using file sets for headers instead of ``install(FILES)``
  would be even better (see :command:`target_sources(FILE_SET)`).

  Note that some of the types' built-in defaults use the ``DATAROOT`` directory as
  a prefix. The ``DATAROOT`` prefix is calculated similarly to the types, with
  ``CMAKE_INSTALL_DATAROOTDIR`` as the variable and ``share`` as the built-in
  default. You cannot use ``DATAROOT`` as a ``TYPE`` parameter; please use
  ``DATA`` instead.

  To make packages compliant with distribution filesystem layout policies, if
  projects must specify a ``DESTINATION``, it is strongly recommended that they use
  a path that begins with the appropriate relative :module:`GNUInstallDirs` variable.
  This allows package maintainers to control the install destination by setting
  the appropriate cache variables.  The following example shows how to follow
  this advice while installing an image to a project-specific documentation
  subdirectory:

  .. code-block:: cmake

    include(GNUInstallDirs)
    install(FILES logo.png
            DESTINATION ${CMAKE_INSTALL_DOCDIR}/myproj
    )

  .. versionadded:: 3.4
    An install destination given as a ``DESTINATION`` argument may
    use "generator expressions" with the syntax ``$<...>``.  See the
    :manual:`cmake-generator-expressions(7)` manual for available expressions.

  .. versionadded:: 3.20
    An install rename given as a ``RENAME`` argument may
    use "generator expressions" with the syntax ``$<...>``.  See the
    :manual:`cmake-generator-expressions(7)` manual for available expressions.

  .. versionadded:: 3.31
    The ``TYPE`` argument now supports type ``LIBEXEC``.

.. signature::
  install(DIRECTORY <dir>... [...])

  .. note::

    To install a directory sub-tree of headers, consider using file sets
    defined by :command:`target_sources(FILE_SET)` instead. File sets not only
    preserve directory structure, they also associate headers with a target
    and install as part of the target.

  Install the contents of one or more directories:

  .. code-block:: cmake

    install(DIRECTORY <dir>...
            TYPE <type> | DESTINATION <dir>
            [FILE_PERMISSIONS <permission>...]
            [DIRECTORY_PERMISSIONS <permission>...]
            [USE_SOURCE_PERMISSIONS] [OPTIONAL] [MESSAGE_NEVER]
            [CONFIGURATIONS <config>...]
            [COMPONENT <component>] [EXCLUDE_FROM_ALL]
            [FILES_MATCHING]
            [<match-rule> <match-option>...]...
            )

  The ``DIRECTORY`` form installs contents of one or more directories to a
  given destination.  The directory structure is copied verbatim to the
  destination.

  Either a ``TYPE`` or a ``DESTINATION`` must be provided, but not both.
  If no permissions is given, files will be given the default permissions
  specified in the `FILES`_ form of the command, and the directories
  will be given the default permissions specified in the `PROGRAMS`_
  form of the command.

  The options are:

  ``<dir>...``
    The list of directories to be installed.

    The last component of each directory name is appended to the
    destination directory but a trailing slash may be used to avoid
    this because it leaves the last component empty.  Directory names
    given as relative paths are interpreted with respect to the current
    source directory.  If no input directory names are given the
    destination directory will be created but nothing will be installed
    into it.

    .. versionadded:: 3.5
      The source ``<dir>...`` list may use "generator expressions" with the
      syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)`
      manual for available expressions.

  ``TYPE <type>``
    Specifies the generic file type of the files within the listed
    directories being installed.  A destination will then be set
    automatically by taking the corresponding variable from
    :module:`GNUInstallDirs`, or by using a built-in default if that
    variable is not defined.  See the table below for the supported
    file types and their corresponding variables and built-in defaults.
    Projects can provide a ``DESTINATION`` argument instead of a file
    type if they wish to explicitly define the install destination.

    ======================= ================================== =========================
       ``TYPE`` Argument         GNUInstallDirs Variable           Built-In Default
    ======================= ================================== =========================
    ``BIN``                 ``${CMAKE_INSTALL_BINDIR}``        ``bin``
    ``SBIN``                ``${CMAKE_INSTALL_SBINDIR}``       ``sbin``
    ``LIB``                 ``${CMAKE_INSTALL_LIBDIR}``        ``lib``
    ``INCLUDE``             ``${CMAKE_INSTALL_INCLUDEDIR}``    ``include``
    ``SYSCONF``             ``${CMAKE_INSTALL_SYSCONFDIR}``    ``etc``
    ``SHAREDSTATE``         ``${CMAKE_INSTALL_SHARESTATEDIR}`` ``com``
    ``LOCALSTATE``          ``${CMAKE_INSTALL_LOCALSTATEDIR}`` ``var``
    ``RUNSTATE``            ``${CMAKE_INSTALL_RUNSTATEDIR}``   ``<LOCALSTATE dir>/run``
    ``DATA``                ``${CMAKE_INSTALL_DATADIR}``       ``<DATAROOT dir>``
    ``INFO``                ``${CMAKE_INSTALL_INFODIR}``       ``<DATAROOT dir>/info``
    ``LOCALE``              ``${CMAKE_INSTALL_LOCALEDIR}``     ``<DATAROOT dir>/locale``
    ``MAN``                 ``${CMAKE_INSTALL_MANDIR}``        ``<DATAROOT dir>/man``
    ``DOC``                 ``${CMAKE_INSTALL_DOCDIR}``        ``<DATAROOT dir>/doc``
    ``LIBEXEC``             ``${CMAKE_INSTALL_LIBEXECDIR}``    ``libexec``
    ======================= ================================== =========================

    Note that some of the types' built-in defaults use the ``DATAROOT``
    directory as a prefix.  The ``DATAROOT`` prefix is calculated similarly
    to the types, with ``CMAKE_INSTALL_DATAROOTDIR`` as the variable and
    ``share`` as the built-in default.  You cannot use ``DATAROOT`` as a
    ``TYPE`` parameter; please use ``DATA`` instead.

    .. versionadded:: 3.31
      The ``LIBEXEC`` type.

  ``DESTINATION <dir>``
    The destination directory, as documented among the `common options`_.

    To make packages compliant with distribution filesystem layout
    policies, if projects must specify a ``DESTINATION``, it is
    strongly recommended that they use a path that begins with the
    appropriate relative :module:`GNUInstallDirs` variable.
    This allows package maintainers to control the install destination
    by setting the appropriate cache variables.

    .. versionadded:: 3.4
      The destination ``<dir>`` may use "generator expressions" with the
      syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)`
      manual for available expressions.

  ``FILE_PERMISSIONS <permission>...``
    Specify permissions given to files in the destination, where the
    ``<permission>`` names are documented among the `common options`_.

  ``DIRECTORY_PERMISSIONS <permission>...``
    Specify permissions given to directories in the destination, where the
    ``<permission>`` names are documented among the `common options`_.

  ``USE_SOURCE_PERMISSIONS``
    If specified, and ``FILE_PERMISSIONS`` is not, file permissions will
    be copied from the source directory structure.

  ``MESSAGE_NEVER``
    .. versionadded:: 3.1

    Disable file installation status output.

  ``FILES_MATCHING``
    This option may be given before the first ``<match-rule>`` to
    disable installation of files (but not directories) not matched
    by any expression.  For example, the code

    .. code-block:: cmake

      install(DIRECTORY src/ DESTINATION doc/myproj
              FILES_MATCHING PATTERN "*.png")

    will extract and install images from a source tree.

  ``<match-rule> <match-option>...``
    Installation of directories may be controlled with fine granularity
    using rules that match directories or files encountered within input
    directories.  They may be used to apply certain options (see below)
    to a subset of the files and directories encountered.  All files
    and directories are installed whether or not they are matched,
    unless the above ``FILES_MATCHING`` option is given.

    The full path to each input file or directory, with forward slashes,
    may be matched by a ``<match-rule>``:

    ``PATTERN <pattern>``
      Match complete file names using a globbing pattern.  The portion of
      the full path matching the pattern must occur at the end of the file
      name and be preceded by a slash (which is not part of the pattern).

    ``REGEX <regex>``
      Match any portion of the full path of a file with a
      :ref:`regular expression <Regex Specification>`.
      One may use ``/`` and ``$`` to limit matching to the end of a path.

    Each ``<match-rule>`` may be followed by ``<match-option>`` arguments.
    The match options apply to the files or directories matched by the rule.
    The match options are:

    ``EXCLUDE``
      Exclude the matched file or directory from installation.

    ``PERMISSIONS <permission>...``
      Ovrerride the permissions setting for the matched file or directory.

    For example, the code

    .. code-block:: cmake

      install(DIRECTORY icons scripts/ DESTINATION share/myproj
              PATTERN "CVS" EXCLUDE
              PATTERN "scripts/*"
              PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                          GROUP_EXECUTE GROUP_READ)

    will install the ``icons`` directory to ``share/myproj/icons`` and the
    ``scripts`` directory to ``share/myproj``.  The icons will get default
    file permissions, the scripts will be given specific permissions, and any
    ``CVS`` directories will be excluded.

.. signature::
  install(SCRIPT <file> [...])
  install(CODE <code> [...])

  Invoke CMake scripts or code during installation:

  .. code-block:: cmake

    install([[SCRIPT <file>] [CODE <code>]]
            [ALL_COMPONENTS | COMPONENT <component>]
            [EXCLUDE_FROM_ALL] [...])

  The ``SCRIPT`` form will invoke the given CMake script files during
  installation.  If the script file name is a relative path it will be
  interpreted with respect to the current source directory.  The ``CODE``
  form will invoke the given CMake code during installation.  Code is
  specified as a single argument inside a double-quoted string.  For
  example, the code

  .. code-block:: cmake

    install(CODE "message(\"Sample install message.\")")

  will print a message during installation.

  .. versionadded:: 3.21
    When the ``ALL_COMPONENTS`` option is given, the custom installation
    script code will be executed for every component of a component-specific
    installation.  This option is mutually exclusive with the ``COMPONENT``
    option.

  .. versionadded:: 3.14
    ``<file>`` or ``<code>`` may use "generator expressions" with the syntax
    ``$<...>`` (in the case of ``<file>``, this refers to their use in the file
    name, not the file's contents).  See the
    :manual:`cmake-generator-expressions(7)` manual for available expressions.

.. signature::
  install(EXPORT <export-name> [...])

  Install a CMake file exporting targets for dependent projects:

  .. code-block:: cmake

    install(EXPORT <export-name> DESTINATION <dir>
            [NAMESPACE <namespace>] [FILE <name>.cmake]
            [PERMISSIONS <permission>...]
            [CONFIGURATIONS <config>...]
            [CXX_MODULES_DIRECTORY <directory>]
            [EXPORT_LINK_INTERFACE_LIBRARIES]
            [COMPONENT <component>]
            [EXCLUDE_FROM_ALL]
            [EXPORT_PACKAGE_DEPENDENCIES])
    install(EXPORT_ANDROID_MK <export-name> DESTINATION <dir> [...])

  The ``EXPORT`` form generates and installs a CMake file containing code to
  import targets from the installation tree into another project.
  Target installations are associated with the export ``<export-name>``
  using the ``EXPORT`` option of the :command:`install(TARGETS)` signature
  documented above.  The ``NAMESPACE`` option will prepend ``<namespace>`` to
  the target names as they are written to the import file.  By default
  the generated file will be called ``<export-name>.cmake`` but the ``FILE``
  option may be used to specify a different name.  The value given to
  the ``FILE`` option must be a file name with the ``.cmake`` extension.

  If a ``CONFIGURATIONS`` option is given then the file will only be installed
  when one of the named configurations is installed.  Additionally, the
  generated import file will reference only the matching target
  configurations.  See the :variable:`CMAKE_MAP_IMPORTED_CONFIG_<CONFIG>`
  variable to map configurations of dependent projects to the installed
  configurations.  The ``EXPORT_LINK_INTERFACE_LIBRARIES`` keyword, if
  present, causes the contents of the properties matching
  ``(IMPORTED_)?LINK_INTERFACE_LIBRARIES(_<CONFIG>)?`` to be exported, when
  policy :policy:`CMP0022` is ``NEW``.

  .. note::
    The installed ``<export-name>.cmake`` file may come with additional
    per-configuration ``<export-name>-*.cmake`` files to be loaded by
    globbing.  Do not use an export name that is the same as the package
    name in combination with installing a ``<package-name>-config.cmake``
    file or the latter may be incorrectly matched by the glob and loaded.

  When a ``COMPONENT`` option is given, the listed ``<component>`` implicitly
  depends on all components mentioned in the export set. The exported
  ``<name>.cmake`` file will require each of the exported components to be
  present in order for dependent projects to build properly. For example, a
  project may define components ``Runtime`` and ``Development``, with shared
  libraries going into the ``Runtime`` component and static libraries and
  headers going into the ``Development`` component. The export set would also
  typically be part of the ``Development`` component, but it would export
  targets from both the ``Runtime`` and ``Development`` components. Therefore,
  the ``Runtime`` component would need to be installed if the ``Development``
  component was installed, but not vice versa. If the ``Development`` component
  was installed without the ``Runtime`` component, dependent projects that try
  to link against it would have build errors. Package managers, such as APT and
  RPM, typically handle this by listing the ``Runtime`` component as a dependency
  of the ``Development`` component in the package metadata, ensuring that the
  library is always installed if the headers and CMake export file are present.

  .. versionadded:: 3.7
    In addition to cmake language files, the ``EXPORT_ANDROID_MK`` mode may be
    used to specify an export to the android ndk build system.  This mode
    accepts the same options as the normal export mode.  The Android
    NDK supports the use of prebuilt libraries, both static and shared. This
    allows cmake to build the libraries of a project and make them available
    to an ndk build system complete with transitive dependencies, include flags
    and defines required to use the libraries.

  ``CXX_MODULES_DIRECTORY``
    .. versionadded:: 3.28

    Specify a subdirectory to store C++ module information for targets in the
    export set. This directory will be populated with files which add the
    necessary target property information to the relevant targets. Note that
    without this information, none of the C++ modules which are part of the
    targets in the export set will support being imported in consuming targets.

  ``EXPORT_PACKAGE_DEPENDENCIES``
    .. note::

      Experimental. Gated by ``CMAKE_EXPERIMENTAL_EXPORT_PACKAGE_DEPENDENCIES``.

    Specify that :command:`find_dependency` calls should be exported. If this
    argument is specified, CMake examines all targets in the export set and
    gathers their ``INTERFACE`` link targets. If any such targets either were
    found with :command:`find_package` or have the
    :prop_tgt:`EXPORT_FIND_PACKAGE_NAME` property set, and such package
    dependency was not disabled by passing ``ENABLED OFF`` to
    :command:`export(SETUP)`, then a :command:`find_dependency` call is
    written with the target's corresponding package name, a ``REQUIRED``
    argument, and any additional arguments specified by the ``EXTRA_ARGS``
    argument of :command:`export(SETUP)`. Any package dependencies that were
    manually specified by passing ``ENABLED ON`` to :command:`export(SETUP)`
    are also added, even if the exported targets don't depend on any targets
    from them.

    The :command:`find_dependency` calls are written in the following order:

    1. Any package dependencies that were listed in :command:`export(SETUP)`
       are written in the order they were first specified, regardless of
       whether or not they contain ``INTERFACE`` dependencies of the
       exported targets.
    2. Any package dependencies that contain ``INTERFACE`` link dependencies
       of the exported targets and that were never specified in
       :command:`export(SETUP)` are written in the order they were first
       found.

  The ``EXPORT`` form is useful to help outside projects use targets built
  and installed by the current project.  For example, the code

  .. code-block:: cmake

    install(TARGETS myexe EXPORT myproj DESTINATION bin)
    install(EXPORT myproj NAMESPACE mp_ DESTINATION lib/myproj)
    install(EXPORT_ANDROID_MK myproj DESTINATION share/ndk-modules)

  will install the executable ``myexe`` to ``<prefix>/bin`` and code to import
  it in the file ``<prefix>/lib/myproj/myproj.cmake`` and
  ``<prefix>/share/ndk-modules/Android.mk``.  An outside project
  may load this file with the include command and reference the ``myexe``
  executable from the installation tree using the imported target name
  ``mp_myexe`` as if the target were built in its own tree.

.. signature::
  install(PACKAGE_INFO <package-name> [...])

  .. versionadded:: 3.31
  .. note::

    Experimental. Gated by ``CMAKE_EXPERIMENTAL_EXPORT_PACKAGE_INFO``.

  Installs a |CPS|_ file exporting targets for dependent projects:

  .. code-block:: cmake

    install(PACKAGE_INFO <package-name> EXPORT <export-name>
            [PROJECT <project-name>|NO_PROJECT_METADATA]
            [APPENDIX <appendix-name>]
            [DESTINATION <dir>]
            [LOWER_CASE_FILE]
            [VERSION <version>
             [COMPAT_VERSION <version>]
             [VERSION_SCHEMA <string>]]
            [DEFAULT_TARGETS <target>...]
            [DEFAULT_CONFIGURATIONS <config>...]
            [LICENSE <license-string>]
            [DEFAULT_LICENSE <license-string>]
            [DESCRIPTION <description-string>]
            [HOMEPAGE_URL <url-string>]
            [PERMISSIONS <permission>...]
            [CONFIGURATIONS <config>...]
            [COMPONENT <component>]
            [EXCLUDE_FROM_ALL])

  The ``PACKAGE_INFO`` form generates and installs a |CPS| file which describes
  installed targets such that they can be consumed by another project.
  Target installations are associated with the export ``<export-name>``
  using the ``EXPORT`` option of the :command:`install(TARGETS)` signature
  documented above.  Unlike :command:`install(EXPORT)`, this information is not
  expressed in CMake code, and can be consumed by tools other than CMake.  When
  imported into another CMake project, the imported targets will be prefixed
  with ``<package-name>::``.  By default, the generated file will be called
  ``<package-name>[-<appendix-name>].cps``.  If ``LOWER_CASE_FILE`` is given,
  the package name as it appears on disk (in both the file name and install
  destination) will be first converted to lower case.

  If ``DESTINATION`` is not specified, a platform-specific default is used.

  Several options may be used to specify package metadata:

  ``VERSION <version>``
    Version of the package.  The ``<version>`` shall conform to the specified
    schema.  Refer to :ref:`Version Selection (CPS) <cps version selection>`
    for more information on how the package version is used when consumers
    request a package.

  ``COMPAT_VERSION <version>``
    Oldest version for which the package provides compatibility.

    If not specified, ``COMPAT_VERSION`` is implicitly taken to equal the
    package's ``VERSION``, which is to say that no backwards compatibility is
    provided.

  ``VERSION_SCHEMA <schema>``
    The schema that the package's version number(s) (both ``VERSION`` and
    ``COMPAT_VERSION``) follow.  While no schema will be written to the
    ``.cps`` file if this option is not provided, CPS specifies that the schema
    is assumed to be ``simple`` in such cases. Refer to |cps-version_schema|_
    for more details and a list of officially supported schemas, but be aware
    that the specification may include schemas that are not supported by CMake.
    See :ref:`Version Selection (CPS) <cps version selection>` for the list of
    schemas supported by :command:`find_package`.

  ``DEFAULT_TARGETS <target>...``

    Targets to be used if a consumer requests linking to the package name,
    rather than to specific components.

  ``DEFAULT_CONFIGURATIONS <config>...``

    Ordered list of configurations consumers should prefer if no exact match or
    mapping of the consumer's configuration to the package's available
    configurations exists.  If not specified, CMake will fall back to the
    package's available configurations in an unspecified order.

  ``LICENSE <license-string>``
    .. versionadded:: 4.2

    A |SPDX|_ (SPDX) `License Expression`_ that describes the license(s) of the
    project as a whole, including documentation, resources, or other materials
    distributed with the project, in addition to software artifacts.  See the
    SPDX `License List`_ for a list of commonly used licenses and their
    identifiers.

    The license of individual components is taken from the
    :prop_tgt:`SPDX_LICENSE` property of their respective targets.

  ``DEFAULT_LICENSE <license-string>``
    .. versionadded:: 4.2

    A |SPDX|_ (SPDX) `License Expression`_ that describes the license(s) of any
    components which do not otherwise specify their license(s).

  ``DESCRIPTION <description-string>``
    .. versionadded:: 4.1

    An informational description of the project.  It is recommended that this
    description is a relatively short string, usually no more than a few words.

  ``HOMEPAGE_URL <url-string>``
    .. versionadded:: 4.1

    An informational canonical home URL for the project.

  By default, if the specified ``<package-name>`` matches the current CMake
  :variable:`PROJECT_NAME`, package metadata will be inherited from the
  project.  The ``PROJECT <project-name>`` option may be used to specify a
  different project from which to inherit metadata.  If ``NO_PROJECT_METADATA``
  is specified, automatic inheritance of package metadata will be disabled.
  In any case, any metadata values specified in the ``install`` command will
  take precedence.

  If ``APPENDIX`` is specified, rather than generating a top level package
  specification, the specified targets will be exported as an appendix to the
  named package.  Appendices may be used to separate less commonly used targets
  (along with their external dependencies) from the rest of a package.  This
  enables consumers to ignore transitive dependencies for targets that they
  don't use, and also allows a single logical "package" to be composed of
  artifacts produced by multiple build trees.

  Appendices are not permitted to change basic package metadata; therefore,
  none of ``PROJECT``, ``VERSION``, ``COMPAT_VERSION``, ``VERSION_SCHEMA``,
  ``DEFAULT_TARGETS`` or ``DEFAULT_CONFIGURATIONS`` may be specified in
  combination with ``APPENDIX``.  Additionally, it is strongly recommended that
  use of ``LOWER_CASE_FILE`` should be consistent between the main package and
  any appendices.

.. signature::
  install(RUNTIME_DEPENDENCY_SET <set-name> [...])

  .. versionadded:: 3.21

  Installs a runtime dependency set:

  .. code-block:: cmake

    install(RUNTIME_DEPENDENCY_SET <set-name>
            [[LIBRARY|RUNTIME|FRAMEWORK]
             [DESTINATION <dir>]
             [PERMISSIONS <permission>...]
             [CONFIGURATIONS <config>...]
             [COMPONENT <component>]
             [NAMELINK_COMPONENT <component>]
             [OPTIONAL] [EXCLUDE_FROM_ALL]
            ] [...]
            [PRE_INCLUDE_REGEXES <regex>...]
            [PRE_EXCLUDE_REGEXES <regex>...]
            [POST_INCLUDE_REGEXES <regex>...]
            [POST_EXCLUDE_REGEXES <regex>...]
            [POST_INCLUDE_FILES <file>...]
            [POST_EXCLUDE_FILES <file>...]
            [DIRECTORIES <dir>...]
            )

  Installs a runtime dependency set previously created by one or more
  :command:`install(TARGETS)` or :command:`install(IMPORTED_RUNTIME_ARTIFACTS)`
  commands.  The dependencies of targets belonging to a runtime dependency set
  are installed in the ``RUNTIME`` destination and component on DLL platforms,
  and in the ``LIBRARY`` destination and component on non-DLL platforms.
  macOS frameworks are installed in the ``FRAMEWORK`` destination and component.
  Targets built within the build tree will never be installed as runtime
  dependencies, nor will their own dependencies, unless the targets themselves
  are installed with :command:`install(TARGETS)`.

  The generated install script calls :command:`file(GET_RUNTIME_DEPENDENCIES)`
  on the build-tree files to calculate the runtime dependencies. The build-tree
  executable files are passed as the ``EXECUTABLES`` argument, the build-tree
  shared libraries as the ``LIBRARIES`` argument, and the build-tree modules as
  the ``MODULES`` argument. On macOS, if one of the executables is a
  :prop_tgt:`MACOSX_BUNDLE`, that executable is passed as the
  ``BUNDLE_EXECUTABLE`` argument. At most one such bundle executable may be in
  the runtime dependency set on macOS. The :prop_tgt:`MACOSX_BUNDLE` property
  has no effect on other platforms. Note that
  :command:`file(GET_RUNTIME_DEPENDENCIES)` only supports collecting the runtime
  dependencies for Windows, Linux and macOS platforms, so
  ``install(RUNTIME_DEPENDENCY_SET)`` has the same limitation.

  The following sub-arguments are forwarded through as the corresponding
  arguments to :command:`file(GET_RUNTIME_DEPENDENCIES)` (for those that provide
  a non-empty list of directories, regular expressions or files).  They all
  support :manual:`generator expressions <cmake-generator-expressions(7)>`.

  * ``DIRECTORIES <dir>...``
  * ``PRE_INCLUDE_REGEXES <regex>...``
  * ``PRE_EXCLUDE_REGEXES <regex>...``
  * ``POST_INCLUDE_REGEXES <regex>...``
  * ``POST_EXCLUDE_REGEXES <regex>...``
  * ``POST_INCLUDE_FILES <file>...``
  * ``POST_EXCLUDE_FILES <file>...``

.. note::
  This command supersedes the :command:`install_targets` command and
  the :prop_tgt:`PRE_INSTALL_SCRIPT` and :prop_tgt:`POST_INSTALL_SCRIPT`
  target properties.  It also replaces the ``FILES`` forms of the
  :command:`install_files` and :command:`install_programs` commands.
  The processing order of these install rules relative to
  those generated by :command:`install_targets`,
  :command:`install_files`, and :command:`install_programs` commands
  is not defined.

Examples
^^^^^^^^

Example: Install Targets with Per-Artifact Components
"""""""""""""""""""""""""""""""""""""""""""""""""""""

Consider a project that defines targets with different artifact kinds:

.. code-block:: cmake

  add_executable(myExe myExe.c)
  add_library(myStaticLib STATIC myStaticLib.c)
  target_sources(myStaticLib PUBLIC FILE_SET HEADERS FILES myStaticLib.h)
  add_library(mySharedLib SHARED mySharedLib.c)
  target_sources(mySharedLib PUBLIC FILE_SET HEADERS FILES mySharedLib.h)
  set_property(TARGET mySharedLib PROPERTY SOVERSION 1)

We may call :command:`install(TARGETS)` with `\<artifact-kind\>`_ arguments
to specify different options for each kind of artifact:

.. code-block:: cmake

  install(TARGETS
            myExe
            mySharedLib
            myStaticLib
          RUNTIME           # Following options apply to runtime artifacts.
            COMPONENT Runtime
          LIBRARY           # Following options apply to library artifacts.
            COMPONENT Runtime
            NAMELINK_COMPONENT Development
          ARCHIVE           # Following options apply to archive artifacts.
            COMPONENT Development
            DESTINATION lib/static
          FILE_SET HEADERS  # Following options apply to file set HEADERS.
            COMPONENT Development
          )

This will:

* Install ``myExe`` to ``<prefix>/bin``, the default RUNTIME artifact
  destination, as part of the ``Runtime`` component.

* On non-DLL platforms:

  * Install ``libmySharedLib.so.1`` to ``<prefix>/lib``, the default
    LIBRARY artifact destination, as part of the ``Runtime`` component.

  * Install the ``libmySharedLib.so`` "namelink" (symbolic link) to
    ``<prefix>/lib``, the default LIBRARY artifact destination, as part
    of the ``Development`` component.

* On DLL platforms:

  * Install ``mySharedLib.dll`` to ``<prefix>/bin``, the default RUNTIME
    artifact destination, as part of the ``Runtime`` component.

  * Install ``mySharedLib.lib`` to ``<prefix>/lib/static``, the specified
    ARCHIVE artifact destination, as part of the ``Development`` component.

* Install ``myStaticLib`` to ``<prefix>/lib/static``, the specified
  ARCHIVE artifact destination, as part of the ``Development`` component.

* Install ``mySharedLib.h`` and ``myStaticLib.h`` to ``<prefix>/include``,
  the default destination for a file set of type HEADERS, as part of the
  ``Development`` component.

Example: Install Targets to Per-Config Destinations
"""""""""""""""""""""""""""""""""""""""""""""""""""

Each :command:`install(TARGETS)` call installs a given target
:ref:`output artifact <Output Artifacts>` to at most one ``DESTINATION``,
but the install rule itself may be filtered by the ``CONFIGURATIONS`` option.
In order to install to a different destination for each configuration, one
call per configuration is needed.  For example, the code:

.. code-block:: cmake

  install(TARGETS myExe
          CONFIGURATIONS Debug
          RUNTIME
            DESTINATION Debug/bin
          )
  install(TARGETS myExe
          CONFIGURATIONS Release
          RUNTIME
            DESTINATION Release/bin
          )

will install ``myExe`` to ``<prefix>/Debug/bin`` in the Debug configuration,
and to ``<prefix>/Release/bin`` in the Release configuration.

Generated Installation Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

  Use of this feature is not recommended. Please consider using the
  :option:`cmake --install` instead.

The ``install()`` command generates a file, ``cmake_install.cmake``, inside
the build directory, which is used internally by the generated install target
and by CPack. You can also invoke this script manually with
:option:`cmake -P`. This script accepts several variables:

``COMPONENT``
  Set this variable to install only a single CPack component as opposed to all
  of them. For example, if you only want to install the ``Development``
  component, run ``cmake -DCOMPONENT=Development -P cmake_install.cmake``.

``BUILD_TYPE``
  Set this variable to change the build type if you are using a multi-config
  generator. For example, to install with the ``Debug`` configuration, run
  ``cmake -DBUILD_TYPE=Debug -P cmake_install.cmake``.

``DESTDIR``
  This is an environment variable rather than a CMake variable. It allows you
  to change the installation prefix on UNIX systems. See :envvar:`DESTDIR` for
  details.

.. _CPS: https://cps-org.github.io/cps/
.. |CPS| replace:: Common Package Specification

.. _cps-version_schema: https://cps-org.github.io/cps/schema.html#version-schema
.. |cps-version_schema| replace:: ``version_schema``

.. _SPDX: https://spdx.dev/
.. |SPDX| replace:: System Package Data Exchange

.. _License Expression: https://spdx.github.io/spdx-spec/v3.0.1/annexes/spdx-license-expressions/
.. _License List: https://spdx.org/licenses/
