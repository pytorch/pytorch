cmake_pkg_config
----------------

.. versionadded:: 3.31

.. only:: html

   .. contents::

Process pkg-config format package files.

Synopsis
^^^^^^^^

.. parsed-literal::

  cmake_pkg_config(`EXTRACT`_ <package> [<version>] [...])
  cmake_pkg_config(`POPULATE`_ <package> [<version>] [...])
  cmake_pkg_config(`IMPORT`_ <package> [<version>] [...])

Introduction
^^^^^^^^^^^^

This command generates CMake variables and targets from pkg-config format
package files natively, without needing to invoke or even require the presence
of a pkg-config implementation. A ``<package>`` is either an absolute path to a
package file, or a package name to be searched for using the typical pkg-config
search patterns. The optional ``<version>`` string has the same format and
semantics as a pkg-config style version specifier, with the exception that if
no comparison operator is specified ``=`` is assumed.

PkgConfig Targets
^^^^^^^^^^^^^^^^^

``cmake_pkg_config`` may recursively generate target-like names in the global
scope in order to resolve a package ``IMPORT`` or ``POPULATE`` command. These
names take the form of ``@foreign_pkgcfg::[<prefix>_]<package>`` and are exposed
via the :prop_tgt:`INTERFACE_LINK_LIBRARIES` target property of an
``IMPORT``-generated target.

It is not possible to modify or address these pkg-config native targets via
normal target-based commands. Limited control over their generation is possible
via the ``POPULATE`` command, but modification should generally be performed
inside the corresponding package file, not downstream in CMake.

Pkg-config targets are reused across commands. Once a given package name has
been resolved via ``POPULATE`` or ``IMPORT`` (but not ``EXTRACT``), all future
requests for the corresponding package name by those commands will resolve to
the previously generated pkg-config target.

``EXTRACT`` always performs the complete package name lookup in order to allow
searches for multiple installations of the same package in custom dependency
management schemes.

Common Options
^^^^^^^^^^^^^^

There are multiple signatures for this command, and some of the options are
common between them. They are:

``EXACT`` / ``QUIET`` / ``REQUIRED``
  The ``EXACT`` option requests that the version string be matched exactly
  (including empty string, if no version is provided), overriding the typical
  pkg-config version comparison algorithm. This will ignore any comparison
  operator attached to the version string.

  The ``QUIET`` option disables informational messages, including those
  indicating that the package cannot be found if it is not ``REQUIRED``. The
  ``REQUIRED`` option stops processing with an error message if the package
  cannot be found.

``STRICTNESS <mode>``
  Specify how strictly the contents of the package files will be verified during
  parsing and resolution. An invalid file, under the provided strictness mode,
  will cause the command to fail. Possible modes are:

  * ``STRICT``: Closely mirrors the behavior of the original FDO pkg-config.
    Variables and keywords must be unique. Variables must be defined before
    they are used. The Name, Description, and Version keywords must be present.
    The overall structure of the file must be valid and parsable.

  * ``PERMISSIVE``: Closely mirrors the behavior of the pkgconf implementation.
    Duplicate variables are overridden. Duplicate keywords are appended.
    Undefined variables resolve to empty strings. The Name, Description, and
    Version keywords must be present. The overall structure of the file must be
    valid and parsable.

  * ``BEST_EFFORT``: Same behavior as ``PERMISSIVE`` with regards to duplicate
    or uninitialized variables and keywords, but will not fail under any
    conditions. Package files which require BEST_EFFORT will fail validation
    under all other major implementations and should be fixed.

  The default strictness is ``PERMISSIVE``.

``ENV_MODE``
  Specifies which environment variables will be queried when running a given
  command. Possible modes are:

  * ``FDO``: Queries only the original set of ``PKG_CONFIG_*`` environment
    variables used by the freedesktop.org ``pkg-config`` implementation.

  * ``PKGCONF``: Queries the more extensive set of environment variables used
    by the ``pkgconf`` implementation.

  * ``IGNORE``: Ignores the presence, absence, and value of environment
    variables entirely. In all cases an environment variable would be queried
    its treated as defined, but with a value of empty string for the purpose
    of the operation. This does not modify the current environment. For boolean
    environment variables, such as ``PKG_CONFIG_ALLOW_*``, this means they are
    evaluated as truthy.

    ``PKG_CONFIG_SYSROOT_PATH`` is a minor exception. When ``ENV_MODE IGNORE``
    is used, no root path prepending will occur by default and ``pc_sysrootdir``
    remains defaulted to ``/``.

  Target-generating subcommands always ignore flag-filtering environment
  variables. The default environment mode is ``PKGCONF``.

``PC_LIBDIR <path>...``
  Overrides the default search location for package files; also used to derive
  the ``pc_path`` package variable.

  When this option is not provided, the default library directory is the first
  available of the following values:

  #. ``CMAKE_PKG_CONFIG_PC_LIB_DIRS``
  #. The ``PKG_CONFIG_LIBDIR`` environment variable
  #. The output of ``pkg-config --variable pc_path pkg-config``
  #. A platform-dependent default value

``PC_PATH <path>...``
  Overrides the supplemental package file directories which will be prepended
  to the search path; also used to derive the ``pc_path`` package variable.

  When this option is not provided, the default paths are the first available of
  the following values:

  #. ``CMAKE_PKG_CONFIG_PC_PATH``
  #. The ``PKG_CONFIG_PATH`` environment variable
  #. Empty list

``DISABLE_UNINSTALLED <bool>``
  Overrides the search behavior for "uninstalled" package files. These are
  package files with an "-uninstalled" suffix which describe packages integrated
  directly from a build tree.

  Normally such package files have higher priority than "installed" packages.
  When ``DISABLE_UNINSTALLED`` is true, searching for "uninstalled" packages
  is disabled.

  When this option is not provided, the default search behavior is determined
  by the first available of the following values:

  #. ``CMAKE_PKG_CONFIG_DISABLE_UNINSTALLED``
  #. If the ``PKG_CONFIG_DISABLE_UNINSTALLED`` environment variable is defined
     the search is disabled, otherwise it is enabled.

``PC_SYSROOT_DIR <path>``
  Overrides the root path which will be prepended to paths specified by ``-I``
  compile flags and ``-L`` library search locations; also used to derive the
  ``pc_sysrootdir`` package variable.

  When this option is not provided, the default root path is provided by the
  first available of the following values:

  #. ``CMAKE_PKG_CONFIG_SYSROOT_DIR``
  #. The ``PKG_CONFIG_SYSROOT_DIR`` environment variable
  #. If no root path is available, nothing will be prepended to include or
     library directory paths and ``pc_sysrootdir`` will be set to ``/``

``TOP_BUILD_DIR <path>``
  Overrides the top build directory path used to derive the ``pc_top_builddir``
  package variable.

  When this option is not provided, the default top build directory path is
  the first available of the following values:

  #. ``CMAKE_PKG_CONFIG_TOP_BUILD_DIR``
  #. The ``PKG_CONFIG_TOP_BUILD_DIR`` environment variable
  #. If no top build directory path is available, the ``pc_top_builddir``
     package variable is not set

``PREFIX <name>``
  Specifying a prefix creates an independent collection of pkg-config targets
  separate from previously populated targets. This enables multiple version of
  a given package to co-exist, for example packages from different sysroots.

  The default prefix is an empty string.

``BIND_PC_REQUIRES <<name>=<target>>...``
  A list of ``<name>=<target>`` pairs, the ``name`` is a package name as it
  appears in the ``Requires`` list of a pkg-config file and the ``target`` is a
  CMake-native target name (not a pkg-config target).

  When a given package name appears in the ``Requires`` list of a package, it
  will be fulfilled with the associated CMake target. This behavior applies to
  all dependencies in the pkg-config graph that have not been previously
  populated.

Signatures
^^^^^^^^^^

.. signature::
  cmake_pkg_config(EXTRACT <package> [<version>] [...])

  .. versionadded:: 3.31

  Extract the contents of the package into variables.

  .. code-block:: cmake

    cmake_pkg_config(EXTRACT <package> [<version>]
                    [REQUIRED] [EXACT] [QUIET]
                    [SYSTEM_INCLUDE_DIRS <path>...]
                    [SYSTEM_LIBRARY_DIRS <path>...]
                    [ALLOW_SYSTEM_INCLUDES <bool>]
                    [ALLOW_SYSTEM_LIBS <bool>]
                    [STRICTNESS <mode>]
                    [ENV_MODE <mode>]
                    [PC_LIBDIR <path>...]
                    [PC_PATH <path>...]
                    [DISABLE_UNINSTALLED <bool>]
                    [PC_SYSROOT_DIR <path>]
                    [TOP_BUILD_DIR <path>])

The following variables will be populated from the contents of package file:

==================================== ====== ========================================================================================
              Variable                Type                       Definition
==================================== ====== ========================================================================================
``CMAKE_PKG_CONFIG_NAME``            String Value of the ``Name`` keyword
``CMAKE_PKG_CONFIG_DESCRIPTION``     String Value of the ``Description`` keyword
``CMAKE_PKG_CONFIG_VERSION``         String Value of the ``Version`` keyword
``CMAKE_PKG_CONFIG_PROVIDES``        List   Value of the ``Provides`` keyword
``CMAKE_PKG_CONFIG_REQUIRES``        List   Value of the ``Requires`` keyword
``CMAKE_PKG_CONFIG_CONFLICTS``       List   Value of the ``Conflicts`` keyword
``CMAKE_PKG_CONFIG_CFLAGS``          String Value of the ``CFlags`` / ``Cflags`` keyword
``CMAKE_PKG_CONFIG_INCLUDES``        List   All ``-I`` prefixed flags from ``CMAKE_PKG_CONFIG_CFLAGS``
``CMAKE_PKG_CONFIG_COMPILE_OPTIONS`` List   All flags not prefixed with ``-I`` from ``CMAKE_PKG_CONFIG_CFLAGS``
``CMAKE_PKG_CONFIG_LIBS``            String Value of the ``Libs`` keyword
``CMAKE_PKG_CONFIG_LIBDIRS``         List   All ``-L`` prefixed flags from ``CMAKE_PKG_CONFIG_LIBS``
``CMAKE_PKG_CONFIG_LIBNAMES``        List   All ``-l`` prefixed flags from ``CMAKE_PKG_CONFIG_LIBS``
``CMAKE_PKG_CONFIG_LINK_OPTIONS``    List   All flags not prefixed with ``-L`` or ``-l`` from ``CMAKE_PKG_CONFIG_LIBS``
``CMAKE_PKG_CONFIG_*_PRIVATE``       \*     ``CFLAGS`` / ``LIBS`` / ``REQUIRES`` and derived, but in their ``.private`` suffix forms
==================================== ====== ========================================================================================

``SYSTEM_INCLUDE_DIRS``
  Overrides the "system" directories for the purpose of flag mangling include
  directories in ``CMAKE_PKG_CONFIG_CFLAGS`` and derived variables.

  When this option is not provided, the default directories are provided by the
  first available of the following values:

  #. ``CMAKE_PKG_CONFIG_SYS_INCLUDE_DIRS``
  #. The ``PKG_CONFIG_SYSTEM_INCLUDE_PATH`` environment variable
  #. The output of ``pkgconf --variable pc_system_includedirs pkg-config``
  #. A platform-dependent default value

  Additionally, when the ``ENV_MODE`` is ``PKGCONF`` the
  ``CMAKE_PKG_CONFIG_PKGCONF_INCLUDES`` variable will be concatenated to the
  list if available. If it is not available, the following environment variables
  will be queried and concatenated:

  * ``CPATH``
  * ``C_INCLUDE_PATH``
  * ``CPLUS_INCLUDE_PATH``
  * ``OBJC_INCLUDE_PATH``
  * ``INCLUDE`` (Windows Only)

``SYSTEM_LIBRARY_DIRS``
  Overrides the "system" directories for the purpose of flag mangling library
  directories in ``CMAKE_PKG_CONFIG_LIBS`` and derived variables.

  When this option is not provided, the default directories are provided by the
  first available of the following values:

  #. ``CMAKE_PKG_CONFIG_SYS_LIB_DIRS``
  #. The ``PKG_CONFIG_SYSTEM_LIBRARY_PATH`` environment variable
  #. The output of ``pkgconf --variable pc_system_libdirs pkg-config``
  #. A platform-dependent default value

  Additionally, when the ``ENV_MODE`` is ``PKGCONF`` the
  ``CMAKE_PKG_CONFIG_PKGCONF_LIB_DIRS`` variable will be concatenated to the
  list if available. If it is not available, the ``LIBRARY_PATH`` environment
  variable will be queried and concatenated.

``ALLOW_SYSTEM_INCLUDES``
  Preserves "system" directories during flag mangling of include directories
  in ``CMAKE_PKG_CONFIG_CFLAGS`` and derived variables.

  When this option is not provided, the default value is determined by the first
  available of the following values:

  #. ``CMAKE_PKG_CONFIG_ALLOW_SYS_INCLUDES``
  #. If the ``PKG_CONFIG_ALLOW_SYSTEM_CFLAGS`` environment variable is defined
     the flags are preserved, otherwise they are filtered during flag mangling.


``ALLOW_SYSTEM_LIBS``
  Preserves "system" directories during flag mangling of library directories
  in ``CMAKE_PKG_CONFIG_LIBS`` and derived variables.

  When this option is not provided, the default value is determined by the first
  available of the following values:

  #. ``CMAKE_PKG_CONFIG_ALLOW_SYS_LIBS``
  #. If the ``PKG_CONFIG_ALLOW_SYSTEM_LIBS`` environment variable is defined
     the flags are preserved, otherwise they are filtered during flag mangling.

.. signature::
  cmake_pkg_config(POPULATE <package> [<version>] [...])

  .. versionadded:: 4.1

  Populate a package in the pkg-config target namespace

  .. code-block:: cmake

    cmake_pkg_config(POPULATE <package> [<version>]
                    [REQUIRED] [EXACT] [QUIET]
                    [PREFIX <prefix>]
                    [BIND_PC_REQUIRES <<name>=<target>>...]
                    [STRICTNESS <mode>]
                    [ENV_MODE <mode>]
                    [PC_LIBDIR <path>...]
                    [PC_PATH <path>...]
                    [DISABLE_UNINSTALLED <bool>]
                    [PC_SYSROOT_DIR <path>]
                    [TOP_BUILD_DIR <path>])

``POPULATE`` enables manual control of resolution of a given package's
``Requires`` list without importing onto a native CMake target. Once populated,
a package and its dependencies will be used for resolution of all future
``POPULATE`` and ``IMPORT`` commands.

A ``PKGCONFIG_<package>_FOUND`` variable will be set to indicate whether the
package was found.

.. signature::
  cmake_pkg_config(IMPORT <package> [<version>] [...])

  .. versionadded:: 4.1

  Import a pkg-config target as a CMake :prop_tgt:`IMPORTED` target

  .. code-block:: cmake

    cmake_pkg_config(IMPORT <package> [<version>]
                    [REQUIRED] [EXACT] [QUIET]
                    [NAME <name>]
                    [PREFIX <prefix>]
                    [BIND_PC_REQUIRES <<name>=<target>>...]
                    [STRICTNESS <mode>]
                    [ENV_MODE <mode>]
                    [PC_LIBDIR <path>...]
                    [PC_PATH <path>...]
                    [DISABLE_UNINSTALLED <bool>]
                    [PC_SYSROOT_DIR <path>]
                    [TOP_BUILD_DIR <path>])

Creates a native CMake ``IMPORTED`` target that can be linked to via
:command:`target_link_libraries`. This new target is named
``PkgConfig::<package>`` by default.

A ``PKGCONFIG_<package>_FOUND`` variable will be set to indicate whether the
package was found.

``NAME``
  Overrides the name of the created CMake target to ``PkgConfig::<name>``. This
  *does not* affect the ``PKGCONFIG_<package>_FOUND`` variable.
