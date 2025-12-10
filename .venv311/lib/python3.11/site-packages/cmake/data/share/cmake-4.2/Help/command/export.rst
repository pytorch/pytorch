export
------

Export targets or packages for outside projects to use them directly
from the current project's build tree, without installation.

See the :command:`install(EXPORT)` command to export targets from an
install tree.

Synopsis
^^^^^^^^

.. parsed-literal::

  export(`TARGETS`_ <target>... [...])
  export(`EXPORT`_ <export-name> [...])
  export(`PACKAGE`_ <PackageName>)
  export(`SETUP`_ <export-name> [...])

Exporting Targets
^^^^^^^^^^^^^^^^^

.. signature::
  export(TARGETS <target>... [...])

.. code-block:: cmake

  export(TARGETS <target>... [NAMESPACE <namespace>]
         [APPEND] FILE <filename> [EXPORT_LINK_INTERFACE_LIBRARIES]
         [CXX_MODULES_DIRECTORY <directory>])

Creates a file ``<filename>`` that may be included by outside projects to
import targets named by ``<target>...`` from the current project's build tree.
This is useful during cross-compiling to build utility executables that can
run on the host platform in one project and then import them into another
project being compiled for the target platform.

The file created by this command is specific to the build tree and
should never be installed.  See the :command:`install(EXPORT)` command to
export targets from an install tree.

The options are:

``NAMESPACE <namespace>``
  Prepend the ``<namespace>`` string to all target names written to the file.

``APPEND``
  Append to the file instead of overwriting it.  This can be used to
  incrementally export multiple targets to the same file.

``EXPORT_LINK_INTERFACE_LIBRARIES``
  Include the contents of the properties named with the pattern
  ``(IMPORTED_)?LINK_INTERFACE_LIBRARIES(_<CONFIG>)?``
  in the export, even when policy :policy:`CMP0022` is NEW.  This is useful
  to support consumers using CMake versions older than 2.8.12.

``CXX_MODULES_DIRECTORY <directory>``
  .. versionadded:: 3.28

  Export C++ module properties to files under the given directory. Each file
  will be named according to the target's export name (without any namespace).
  These files will automatically be included from the export file.

This signature requires all targets to be listed explicitly.  If a library
target is included in the export, but a target to which it links is not
included, the behavior is unspecified.  See the :command:`export(EXPORT)` signature
to automatically export the same targets from the build tree as
:command:`install(EXPORT)` would from an install tree.

.. note::

  :ref:`Object Libraries` under :generator:`Xcode` have special handling if
  multiple architectures are listed in :variable:`CMAKE_OSX_ARCHITECTURES`.
  In this case they will be exported as :ref:`Interface Libraries` with
  no object files available to clients.  This is sufficient to satisfy
  transitive usage requirements of other targets that link to the
  object libraries in their implementation.

This command exports all :ref:`build configurations` from the build tree.
See the :variable:`CMAKE_MAP_IMPORTED_CONFIG_<CONFIG>` variable to map
configurations of dependent projects to the exported configurations.

Exporting Targets to Android.mk
"""""""""""""""""""""""""""""""

.. code-block:: cmake

  export(TARGETS <target>... ANDROID_MK <filename>)

.. versionadded:: 3.7

This signature exports CMake built targets to the android ndk build system
by creating an ``Android.mk`` file that references the prebuilt targets. The
Android NDK supports the use of prebuilt libraries, both static and shared.
This allows CMake to build the libraries of a project and make them available
to an ndk build system complete with transitive dependencies, include flags
and defines required to use the libraries. The signature takes a list of
targets and puts them in the ``Android.mk`` file specified by the
``<filename>`` given. This signature can only be used if policy
:policy:`CMP0022` is NEW for all targets given. A error will be issued if
that policy is set to OLD for one of the targets.

Exporting Targets matching install(EXPORT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. signature::
  export(EXPORT <export-name> [...])

.. code-block:: cmake

  export(EXPORT <export-name> [NAMESPACE <namespace>] [FILE <filename>]
         [CXX_MODULES_DIRECTORY <directory>] [EXPORT_PACKAGE_DEPENDENCIES])

Creates a file ``<filename>`` that may be included by outside projects to
import targets from the current project's build tree.  This is the same
as the :command:`export(TARGETS)` signature, except that the targets are not
explicitly listed.  Instead, it exports the targets associated with
the installation export ``<export-name>``.  Target installations may be
associated with the export ``<export-name>`` using the ``EXPORT`` option
of the :command:`install(TARGETS)` command.

``EXPORT_PACKAGE_DEPENDENCIES``
  .. note::

    Experimental. Gated by ``CMAKE_EXPERIMENTAL_EXPORT_PACKAGE_DEPENDENCIES``.

  Specify that :command:`find_dependency` calls should be exported. See
  :command:`install(EXPORT)` for details on how this works.

Exporting Targets to the |CPS|
""""""""""""""""""""""""""""""

.. code-block:: cmake

  export(EXPORT <export-name> PACKAGE_INFO <package-name>
         [PROJECT <project-name>|NO_PROJECT_METADATA]
         [APPENDIX <appendix-name>]
         [LOWER_CASE_FILE]
         [VERSION <version>
          [COMPAT_VERSION <version>]
          [VERSION_SCHEMA <string>]]
         [DEFAULT_TARGETS <target>...]
         [DEFAULT_CONFIGURATIONS <config>...]
         [LICENSE <license-string>]
         [DEFAULT_LICENSE <license-string>]
         [DESCRIPTION <description-string>]
         [HOMEPAGE_URL <url-string>])

.. versionadded:: 4.1
.. note::

  Experimental. Gated by ``CMAKE_EXPERIMENTAL_EXPORT_PACKAGE_INFO``.

Creates a file in the |CPS|_ that may be included by outside projects to import
targets named by ``<target>...`` from the current project's build tree.  See
the :command:`install(PACKAGE_INFO)` command to export targets from an install
tree.  The imported targets are implicitly in the namespace ``<package-name>``.

The default file name is ``<package-name>[-<appendix-name>].cps``. If the
``LOWER_CASE_FILE`` option is given, the file name will use the package name
converted to lower case.

See :command:`install(PACKAGE_INFO)` for a description of the other options.

Exporting Packages
^^^^^^^^^^^^^^^^^^

.. signature::
  export(PACKAGE <PackageName>)

.. code-block:: cmake

  export(PACKAGE <PackageName>)

Store the current build directory in the CMake user package registry
for package ``<PackageName>``.  The :command:`find_package` command may consider the
directory while searching for package ``<PackageName>``.  This helps dependent
projects find and use a package from the current project's build tree
without help from the user.  Note that the entry in the package
registry that this command creates works only in conjunction with a
package configuration file (``<PackageName>Config.cmake``) that works with the
build tree. In some cases, for example for packaging and for system
wide installations, it is not desirable to write the user package
registry.

.. versionchanged:: 3.1
  If the :variable:`CMAKE_EXPORT_NO_PACKAGE_REGISTRY` variable
  is enabled, the ``export(PACKAGE)`` command will do nothing.

.. versionchanged:: 3.15
  By default the ``export(PACKAGE)`` command does nothing (see policy
  :policy:`CMP0090`) because populating the user package registry has effects
  outside the source and build trees.  Set the
  :variable:`CMAKE_EXPORT_PACKAGE_REGISTRY` variable to add build directories
  to the CMake user package registry.

Configuring Exports
^^^^^^^^^^^^^^^^^^^

.. signature::
  export(SETUP <export-name> [...])

.. code-block:: cmake

  export(SETUP <export-name>
         [PACKAGE_DEPENDENCY <dep>
          [ENABLED (<bool-true>|<bool-false>|AUTO)]
          [EXTRA_ARGS <args>...]
         ] [...]
         [TARGET <target>
          [XCFRAMEWORK_LOCATION <location>]
         ] [...]
         )

.. versionadded:: 3.29

Configure the parameters of an export. The arguments are as follows:

``PACKAGE_DEPENDENCY <dep>``
  .. note::

    Experimental. Gated by ``CMAKE_EXPERIMENTAL_EXPORT_PACKAGE_DEPENDENCIES``.

  Specify a package dependency to configure. This changes how
  :command:`find_dependency` calls are written during
  :command:`export(EXPORT)` and :command:`install(EXPORT)`. ``<dep>`` is the
  name of a package to export. This argument accepts the following additional
  arguments:

  ``ENABLED``
    Manually control whether or not the dependency is exported. This accepts
    the following values:

    ``<bool-true>``
      Any value that CMake recognizes as "true". Always export the dependency,
      even if no exported targets depend on it. This can be used to manually
      add :command:`find_dependency` calls to the export.

    ``<bool-false>``
      Any value that CMake recognizes as "false". Never export the dependency,
      even if an exported target depends on it.

    ``AUTO``
      Only export the dependency if an exported target depends on it.

  ``EXTRA_ARGS <args>``
    Specify additional arguments to pass to :command:`find_dependency` after
    the ``REQUIRED`` argument.

``TARGET <target>``
  Specify a target to configure in this export. This argument accepts the
  following additional arguments:

  ``XCFRAMEWORK_LOCATION``
    Specify the location of an ``.xcframework`` which contains the library from
    this target. If specified, the generated code will check to see if the
    ``.xcframework`` exists, and if it does, it will use the ``.xcframework``
    as its imported location instead of the installed library.

.. _CPS: https://cps-org.github.io/cps/
.. |CPS| replace:: Common Package Specification
