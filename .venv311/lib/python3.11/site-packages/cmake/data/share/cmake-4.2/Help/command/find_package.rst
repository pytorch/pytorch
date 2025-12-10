find_package
------------

.. |FIND_XXX| replace:: find_package
.. |FIND_ARGS_XXX| replace:: <PackageName>
.. |FIND_XXX_REGISTRY_VIEW_DEFAULT| replace:: ``TARGET``
.. |CMAKE_FIND_ROOT_PATH_MODE_XXX| replace::
   :variable:`CMAKE_FIND_ROOT_PATH_MODE_PACKAGE`

.. only:: html

   .. contents::

.. note:: The :guide:`Using Dependencies Guide` provides a high-level
  introduction to this general topic. It provides a broader overview of
  where the ``find_package()`` command fits into the bigger picture,
  including its relationship to the :module:`FetchContent` module.
  The guide is recommended pre-reading before moving on to the details below.

Find a package (usually provided by something external to the project),
and load its package-specific details.  Calls to this command can also
be intercepted by :ref:`dependency providers <dependency_providers>`.

Typical Usage
^^^^^^^^^^^^^

Most calls to ``find_package()`` typically have the following form:

.. code-block:: cmake

  find_package(<PackageName> [<version>] [REQUIRED] [COMPONENTS <components>...])

The ``<PackageName>`` is the only mandatory argument.  The ``<version>`` is
often omitted, and ``REQUIRED`` should be given if the project cannot be
configured successfully without the package.  Some more complicated packages
support components which can be selected with the ``COMPONENTS`` keyword, but
most packages don't have that level of complexity.

The above is a reduced form of the `basic signature`_.  Where possible,
projects should find packages using this form.  This reduces complexity and
maximizes the ways in which the package can be found or provided.

Understanding the `basic signature`_ should be enough for general usage of
``find_package()``.  Project maintainers who intend to provide a package
configuration file should understand the bigger picture, as explained in
:ref:`Full Signature` and all subsequent sections on this page.

Search Modes
^^^^^^^^^^^^

The command has a few modes by which it searches for packages:

.. _`Module mode`:

**Module mode**
  In this mode, CMake searches for a file called ``Find<PackageName>.cmake``,
  looking first in the locations listed in the :variable:`CMAKE_MODULE_PATH`,
  then among the :ref:`Find Modules` provided by the CMake installation.
  If the file is found, it is read and processed by CMake.  It is responsible
  for finding the package, checking the version, and producing any needed
  messages.  Some Find modules provide limited or no support for versioning;
  check the Find module's documentation.

  The ``Find<PackageName>.cmake`` file is not typically provided by the
  package itself.  Rather, it is normally provided by something external to
  the package, such as the operating system, CMake itself, or even the project
  from which the ``find_package()`` command was called.  Being externally
  provided, :ref:`Find Modules` tend to be heuristic in nature and are
  susceptible to becoming out-of-date.  They typically search for certain
  libraries, files and other package artifacts.

  Module mode is only supported by the
  :ref:`basic command signature <Basic Signature>`.

.. _`Config mode`:

**Config mode**
  In this mode, CMake searches for a file called
  ``<lowercasePackageName>-config.cmake`` or ``<PackageName>Config.cmake``.
  It will also look for ``<lowercasePackageName>-config-version.cmake`` or
  ``<PackageName>ConfigVersion.cmake`` if version details were specified
  (see :ref:`version selection` for an explanation of how these separate
  version files are used).

  .. note::
    If the experimental ``CMAKE_EXPERIMENTAL_FIND_CPS_PACKAGES`` is enabled,
    files named ``<PackageName>.cps`` and ``<lowercasePackageName>.cps`` are
    also considered.  These files provide package information according to the
    |CPS|_ (CPS), which is more portable than CMake script.  Aside from any
    explicitly noted exceptions, any references to "config files", "config
    mode", "package configuration files", and so forth refer equally to both
    CPS and CMake-script files.  This functionality is a work in progress, and
    some features may be missing.

    Search is implemented in a manner that will tend to prefer |CPS| files
    over CMake-script config files in most cases.  Specifying ``CONFIGS``
    suppresses consideration of CPS files.

  In config mode, the command can be given a list of names to search for
  as package names.  The locations where CMake searches for the config and
  version files is considerably more complicated than for Module mode
  (see :ref:`search procedure`).

  The config and version files are typically installed as part of the
  package, so they tend to be more reliable than Find modules.  They usually
  contain direct knowledge of the package contents, so no searching or
  heuristics are needed within the config or version files themselves.

  Config mode is supported by both the :ref:`basic <Basic Signature>` and
  :ref:`full <Full Signature>` command signatures.

**FetchContent redirection mode**
  .. versionadded:: 3.24
    A call to ``find_package()`` can be redirected internally to a package
    provided by the :module:`FetchContent` module.  To the caller, the behavior
    will appear similar to Config mode, except that the search logic is
    by-passed and the component information is not used.  See
    :command:`FetchContent_Declare` and :command:`FetchContent_MakeAvailable`
    for further details.

When not redirected to a package provided by :module:`FetchContent`, the
command arguments determine whether Module or Config mode is used.  When the
`basic signature`_ is used, the command searches in Module mode first.
If the package is not found, the search falls back to Config mode.
A user may set the :variable:`CMAKE_FIND_PACKAGE_PREFER_CONFIG` variable
to true to reverse the priority and direct CMake to search using Config mode
first before falling back to Module mode.  The basic signature can also be
forced to use only Module mode with a ``MODULE`` keyword.  If the
`full signature`_ is used, the command only searches in Config mode.

.. _`basic signature`:

Basic Signature
^^^^^^^^^^^^^^^

.. code-block:: cmake

  find_package(<PackageName> [version] [EXACT] [QUIET] [MODULE]
               [REQUIRED|OPTIONAL] [[COMPONENTS] [components...]]
               [OPTIONAL_COMPONENTS components...]
               [REGISTRY_VIEW  (64|32|64_32|32_64|HOST|TARGET|BOTH)]
               [GLOBAL]
               [NO_POLICY_SCOPE]
               [BYPASS_PROVIDER]
               [UNWIND_INCLUDE])

The basic signature is supported by both Module and Config modes.
The ``MODULE`` keyword implies that only Module mode can be used to find
the package, with no fallback to Config mode.

Regardless of the mode used, a ``<PackageName>_FOUND`` variable will be
set to indicate whether the package was found.  When the package is found,
package-specific information may be provided through other variables and
:ref:`Imported Targets` documented by the package itself.  The
``QUIET`` option disables informational messages, including those indicating
that the package cannot be found if it is not ``REQUIRED``.  The ``REQUIRED``
option stops processing with an error message if the package cannot be found.

A package-specific list of required components may be listed after the
``COMPONENTS`` keyword.  If any of these components are not able to be
satisfied, the package overall is considered to be not found.  If the
``REQUIRED`` option is also present, this is treated as a fatal error,
otherwise execution still continues.  As a form of shorthand, if the
``REQUIRED`` option is present, the ``COMPONENTS`` keyword can be omitted
and the required components can be listed directly after ``REQUIRED``.

The :variable:`CMAKE_FIND_REQUIRED` variable can be enabled to make this call
``REQUIRED`` by default. This behavior can be overridden by providing the
``OPTIONAL`` keyword. As with the ``REQUIRED`` option, a list of components
can be listed directly after ``OPTIONAL``, which is equivalent to listing
them after the ``COMPONENTS`` keyword. When the ``OPTIONAL`` keyword is given,
the warning output when a package is not found is suppressed.

Additional optional components may be listed after ``OPTIONAL_COMPONENTS``.
If these cannot be satisfied, the package overall can still be considered
found, as long as all required components are satisfied.

The set of available components and their meaning are defined by the
target package:

* For CMake-script package configuration files, it is formally up to the target
  package how to interpret the component information given to it, but it should
  follow the expectations stated above.  For calls where no components are
  specified, there is no single expected behavior and target packages should
  clearly define what occurs in such cases.  Common arrangements include
  assuming it should find all components, no components or some well-defined
  subset of the available components.

* |CPS| packages consist of a root configuration file and zero or more
  appendices, each of which provide components and may have dependencies.
  CMake always attempts to load the root configuration file.  Appendices are
  only loaded if their dependencies can be satisfied, and if they either
  provide requested components, or if no components were requested.  If the
  dependencies of an appendix providing a required component cannot be
  satisfied, the package is considered not found.  Otherwise, that appendix
  is ignored.

.. versionadded:: 3.24
  The ``REGISTRY_VIEW`` keyword specifies which registry views should be
  queried. This keyword is only meaningful on ``Windows`` platforms and will
  be ignored on all others. Formally, it is up to the target package how to
  interpret the registry view information given to it.

.. versionadded:: 3.24
  Specifying the ``GLOBAL`` keyword will promote all imported targets to
  a global scope in the importing project. Alternatively, this functionality
  can be enabled by setting the :variable:`CMAKE_FIND_PACKAGE_TARGETS_GLOBAL`
  variable.

.. _FIND_PACKAGE_VERSION_FORMAT:

The ``[version]`` argument requests a version with which the package found
should be compatible. There are two possible forms in which it may be
specified:

* A single version with the format ``major[.minor[.patch[.tweak]]]``, where
  each component is a numeric value.
* A version range with the format ``versionMin...[<]versionMax`` where
  ``versionMin`` and ``versionMax`` have the same format and constraints on
  components being integers as the single version.  By default, both end points
  are included.  By specifying ``<``, the upper end point will be excluded.
  Version ranges are only supported with CMake 3.19 or later.

.. note::
  With the exception of CPS packages, version support is currently provided
  only on a package-by-package basis.  When a version range is specified but
  the package is only designed to expect a single version, the package will
  ignore the upper end point of the range and only take the single version at
  the lower end of the range into account.  Non-CPS packages that do support
  version ranges do so in a manner that is determined by the individual
  package.  See the `Version Selection`_ section below for details and
  important caveats.

The ``EXACT`` option requests that the version be matched exactly. This option
is incompatible with the specification of a version range.

If no ``[version]`` and/or component list is given to a recursive invocation
inside a find-module, the corresponding arguments are forwarded
automatically from the outer call (including the ``EXACT`` flag for
``[version]``).

See the :command:`cmake_policy` command documentation for discussion
of the ``NO_POLICY_SCOPE`` option.

.. versionadded:: 3.24
  The ``BYPASS_PROVIDER`` keyword is only allowed when ``find_package()`` is
  being called by a :ref:`dependency provider <dependency_providers>`.
  It can be used by providers to call the built-in ``find_package()``
  implementation directly and prevent that call from being re-routed back to
  itself.  Future versions of CMake may detect attempts to use this keyword
  from places other than a dependency provider and halt with a fatal error.

.. versionadded:: 4.2
  The ``UNWIND_INCLUDE`` keyword is only allowed when ``find_package()`` is
  being called within a parent call to ``find_package()``. When a call to
  ``find_package(UNWIND_INCLUDE)`` fails to find the desired package, it begins
  an "unwind" state. In this state further calls to ``find_package()`` and
  :command:`include()` are forbidden, and all parent :command:`include()`
  commands will immediately invoke :command:`return()` when their scope is
  reached. This "unwinding" will continue until the parent ``find_package()``
  is returned to.

  ``UNWIND_INCLUDE`` is only intended to be used by calls to ``find_package()``
  generated by :command:`install(EXPORT_PACKAGE_DEPENDENCIES)`, but may be
  useful to those who wish to manually manage their dependencies in a similar
  manner.

.. _`full signature`:

Full Signature
^^^^^^^^^^^^^^

.. code-block:: cmake

  find_package(<PackageName> [version] [EXACT] [QUIET]
               [REQUIRED|OPTIONAL] [[COMPONENTS] [components...]]
               [OPTIONAL_COMPONENTS components...]
               [CONFIG|NO_MODULE]
               [GLOBAL]
               [NO_POLICY_SCOPE]
               [BYPASS_PROVIDER]
               [NAMES name1 [name2 ...]]
               [CONFIGS config1 [config2 ...]]
               [HINTS path1 [path2 ...]]
               [PATHS path1 [path2 ...]]
               [REGISTRY_VIEW  (64|32|64_32|32_64|HOST|TARGET|BOTH)]
               [PATH_SUFFIXES suffix1 [suffix2 ...]]
               [NO_DEFAULT_PATH]
               [NO_PACKAGE_ROOT_PATH]
               [NO_CMAKE_PATH]
               [NO_CMAKE_ENVIRONMENT_PATH]
               [NO_SYSTEM_ENVIRONMENT_PATH]
               [NO_CMAKE_PACKAGE_REGISTRY]
               [NO_CMAKE_BUILDS_PATH] # Deprecated; does nothing.
               [NO_CMAKE_SYSTEM_PATH]
               [NO_CMAKE_INSTALL_PREFIX]
               [NO_CMAKE_SYSTEM_PACKAGE_REGISTRY]
               [CMAKE_FIND_ROOT_PATH_BOTH |
                ONLY_CMAKE_FIND_ROOT_PATH |
                NO_CMAKE_FIND_ROOT_PATH])

The ``CONFIG`` option, the synonymous ``NO_MODULE`` option, or the use
of options not specified in the `basic signature`_ all enforce pure Config
mode.  In pure Config mode, the command skips Module mode search and
proceeds at once with Config mode search.

Config mode search attempts to locate a configuration file provided by the
package to be found.  A cache entry called ``<PackageName>_DIR`` is created to
hold the directory containing the file.  By default, the command searches for
a package with the name ``<PackageName>``.  If the ``NAMES`` option is given,
the names following it are used instead of ``<PackageName>``.  The names are
also considered when determining whether to redirect the call to a package
provided by :module:`FetchContent`.

The command searches for a file called ``<PackageName>Config.cmake`` or
``<lowercasePackageName>-config.cmake`` for each name specified.
A replacement set of possible configuration file names may be given
using the ``CONFIGS`` option.  The :ref:`search procedure` is specified below.
Once found, any :ref:`version constraint <version selection>` is checked,
and if satisfied, the configuration file is read and processed by CMake.
Since the file is provided by the package it already knows the
location of package contents.  The full path to the configuration file
is stored in the CMake variable ``<PackageName>_CONFIG``.

.. note::

  If the experimental ``CMAKE_EXPERIMENTAL_FIND_CPS_PACKAGES`` is enabled,
  files named ``<PackageName>.cps`` and ``<lowercasePackageName>.cps`` are
  also considered, unless ``CONFIGS`` is given.

All configuration files which have been considered by CMake while
searching for the package with an appropriate version are stored in the
``<PackageName>_CONSIDERED_CONFIGS`` variable, and the associated versions
in the ``<PackageName>_CONSIDERED_VERSIONS`` variable.

If the package configuration file cannot be found, CMake will generate
an error describing the problem unless the ``QUIET`` argument is
specified.  If ``REQUIRED`` is specified and the package is not found, a
fatal error is generated and the configure step stops executing.  If
``<PackageName>_DIR`` has been set to a directory not containing a
configuration file, or if the requested version is not compatible
with the package found in that directory (see :ref:`version selection`),
CMake will ignore it and search from scratch.

Package maintainers providing package configuration files are encouraged to
name and install them such that the :ref:`search procedure` outlined below
will find them without requiring use of additional options.

.. _`search procedure`:

Config Mode Search Procedure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
  When Config mode is used, this search procedure is applied regardless of
  whether the :ref:`full <full signature>` or :ref:`basic <basic signature>`
  signature was given.

.. versionadded:: 3.24
  All calls to ``find_package()`` (even in Module mode) first look for a config
  package file in the :variable:`CMAKE_FIND_PACKAGE_REDIRECTS_DIR` directory.
  The :module:`FetchContent` module, or even the project itself, may write files
  to that location to redirect ``find_package()`` calls to content already
  provided by the project.  If no config package file is found in that location,
  the search proceeds with the logic described below.

CMake constructs a set of possible installation prefixes for the
package.  Under each prefix several directories are searched for a
configuration file.  The tables below show the directories searched.
Each entry is meant for installation trees following Windows (``W``), UNIX
(``U``), or Apple (``A``) conventions:

==================================================================== ==========
 Entry                                                               Convention
==================================================================== ==========
 ``<prefix>/<name>/cps/`` [#p2]_                                        W
 ``<prefix>/<name>/*/cps/`` [#p2]_                                      W
 ``<prefix>/cps/<name>/`` [#p2]_                                        W
 ``<prefix>/cps/<name>/*/`` [#p2]_                                      W
 ``<prefix>/cps/`` [#p2]_                                               W
 ``<prefix>/``                                                          W
 ``<prefix>/(cmake|CMake)/``                                            W
 ``<prefix>/<name>*/``                                                  W
 ``<prefix>/<name>*/(cmake|CMake)/``                                    W
 ``<prefix>/<name>*/(cmake|CMake)/<name>*/`` [#p1]_                     W
 ``<prefix>/(lib/<arch>|lib*|share)/cps/<name>/`` [#p2]_                U
 ``<prefix>/(lib/<arch>|lib*|share)/cps/<name>/*/`` [#p2]_              U
 ``<prefix>/(lib/<arch>|lib*|share)/cps/`` [#p2]_                       U
 ``<prefix>/(lib/<arch>|lib*|share)/cmake/<name>*/``                    U
 ``<prefix>/(lib/<arch>|lib*|share)/<name>*/``                          U
 ``<prefix>/(lib/<arch>|lib*|share)/<name>*/(cmake|CMake)/``            U
 ``<prefix>/<name>*/(lib/<arch>|lib*|share)/cmake/<name>*/``            W/U
 ``<prefix>/<name>*/(lib/<arch>|lib*|share)/<name>*/``                  W/U
 ``<prefix>/<name>*/(lib/<arch>|lib*|share)/<name>*/(cmake|CMake)/``    W/U
==================================================================== ==========

.. [#p1] .. versionadded:: 3.25

.. [#p2] .. versionadded:: 4.0

On systems supporting macOS :prop_tgt:`FRAMEWORK` and :prop_tgt:`BUNDLE`, the
following directories are searched for Frameworks or Application Bundles
containing a configuration file:

=============================================================== ==========
 Entry                                                          Convention
=============================================================== ==========
 ``<prefix>/<name>.framework/Versions/*/Resources/CPS/`` [#p3]_    A
 ``<prefix>/<name>.framework/Resources/CPS/`` [#p3]_               A
 ``<prefix>/<name>.framework/Resources/``                          A
 ``<prefix>/<name>.framework/Resources/CMake/``                    A
 ``<prefix>/<name>.framework/Versions/*/Resources/``               A
 ``<prefix>/<name>.framework/Versions/*/Resources/CMake/``         A
 ``<prefix>/<name>.app/Contents/Resources/CPS/`` [#p3]_            A
 ``<prefix>/<name>.app/Contents/Resources/``                       A
 ``<prefix>/<name>.app/Contents/Resources/CMake/``                 A
=============================================================== ==========

.. [#p3] .. versionadded:: 4.0

When searching the above paths, ``find_package`` will only look for ``.cps``
files in search paths which contain ``/cps/``, and will only look for
``.cmake`` files otherwise.  (This only applies to the paths as specified and
does not consider the contents of ``<prefix>`` or ``<name>``.)

In all cases the ``<name>`` is treated as case-insensitive and corresponds
to any of the names specified (``<PackageName>`` or names given by ``NAMES``).

If at least one compiled language has been enabled, the architecture-specific
``lib/<arch>`` and ``lib*`` directories may be searched based on the compiler's
target architecture, in the following order:

``lib/<arch>``
  Searched if the :variable:`CMAKE_LIBRARY_ARCHITECTURE` variable is set.

``lib64``
  Searched on 64 bit platforms (:variable:`CMAKE_SIZEOF_VOID_P` is 8) and the
  :prop_gbl:`FIND_LIBRARY_USE_LIB64_PATHS` property is set to ``TRUE``.

``lib32``
  Searched on 32 bit platforms (:variable:`CMAKE_SIZEOF_VOID_P` is 4) and the
  :prop_gbl:`FIND_LIBRARY_USE_LIB32_PATHS` property is set to ``TRUE``.

``libx32``
  Searched on platforms using the x32 ABI
  if the :prop_gbl:`FIND_LIBRARY_USE_LIBX32_PATHS` property is set to ``TRUE``.

``lib``
  Always searched.

.. versionchanged:: 3.24
  On ``Windows`` platform, it is possible to include registry queries as part
  of the directories specified through ``HINTS`` and ``PATHS`` keywords, using
  a :ref:`dedicated syntax <Find Using Windows Registry>`. Such specifications
  will be ignored on all other platforms.

.. versionadded:: 3.24
  ``REGISTRY_VIEW`` can be specified to manage ``Windows`` registry queries
  specified as part of ``PATHS`` and ``HINTS``.

  .. include:: include/FIND_XXX_REGISTRY_VIEW.rst

If ``PATH_SUFFIXES`` is specified, the suffixes are appended to each
(``W``) or (``U``) directory entry one-by-one.

This set of directories is intended to work in cooperation with
projects that provide configuration files in their installation trees.
Directories above marked with (``W``) are intended for installations on
Windows where the prefix may point at the top of an application's
installation directory.  Those marked with (``U``) are intended for
installations on UNIX platforms where the prefix is shared by multiple
packages.  This is merely a convention, so all (``W``) and (``U``) directories
are still searched on all platforms.  Directories marked with (``A``) are
intended for installations on Apple platforms.  The
:variable:`CMAKE_FIND_FRAMEWORK` and :variable:`CMAKE_FIND_APPBUNDLE`
variables determine the order of preference.

.. warning::

  Setting :variable:`CMAKE_FIND_FRAMEWORK` or :variable:`CMAKE_FIND_APPBUNDLE`
  to values other than ``FIRST`` (the default) will cause CMake to search for
  |CPS| files in an order that is different from the order set forth in the
  specification.

The set of installation prefixes is constructed using the following
steps.  If ``NO_DEFAULT_PATH`` is specified all ``NO_*`` options are
enabled.

1. Search prefixes unique to the current ``<PackageName>`` being found.
   See policy :policy:`CMP0074`.

   .. versionadded:: 3.12

   Specifically, search prefixes specified by the following variables,
   in order:

   a. :variable:`<PackageName>_ROOT` CMake variable,
      where ``<PackageName>`` is the case-preserved package name.

   b. :variable:`<PACKAGENAME>_ROOT` CMake variable,
      where ``<PACKAGENAME>`` is the upper-cased package name.
      See policy :policy:`CMP0144`.

      .. versionadded:: 3.27

   c. :envvar:`<PackageName>_ROOT` environment variable,
      where ``<PackageName>`` is the case-preserved package name.

   d. :envvar:`<PACKAGENAME>_ROOT` environment variable,
      where ``<PACKAGENAME>`` is the upper-cased package name.
      See policy :policy:`CMP0144`.

      .. versionadded:: 3.27

   The package root variables are maintained as a stack so if
   called from within a find module, root paths from the parent's find
   module will also be searched after paths for the current package.
   This can be skipped if ``NO_PACKAGE_ROOT_PATH`` is passed or by setting
   the :variable:`CMAKE_FIND_USE_PACKAGE_ROOT_PATH` to ``FALSE``.

2. Search paths specified in CMake-specific cache variables.  These
   are intended to be used on the command line with a :option:`-DVAR=VALUE <cmake -D>`.
   The values are interpreted as :ref:`semicolon-separated lists <CMake Language Lists>`.
   This can be skipped if ``NO_CMAKE_PATH`` is passed or by setting the
   :variable:`CMAKE_FIND_USE_CMAKE_PATH` to ``FALSE``:

   * :variable:`CMAKE_PREFIX_PATH`
   * :variable:`CMAKE_FRAMEWORK_PATH`
   * :variable:`CMAKE_APPBUNDLE_PATH`

3. Search paths specified in CMake-specific environment variables.
   These are intended to be set in the user's shell configuration,
   and therefore use the host's native path separator
   (``;`` on Windows and ``:`` on UNIX).
   This can be skipped if ``NO_CMAKE_ENVIRONMENT_PATH`` is passed or by setting
   the :variable:`CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH` to ``FALSE``:

   * ``<PackageName>_DIR``
   * :envvar:`CMAKE_PREFIX_PATH`
   * :envvar:`CMAKE_FRAMEWORK_PATH`
   * :envvar:`CMAKE_APPBUNDLE_PATH`

4. Search paths specified by the ``HINTS`` option.  These should be paths
   computed by system introspection, such as a hint provided by the
   location of another item already found.  Hard-coded guesses should
   be specified with the ``PATHS`` option.

5. Search the standard system environment variables.  This can be
   skipped if ``NO_SYSTEM_ENVIRONMENT_PATH`` is passed  or by setting the
   :variable:`CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH` to ``FALSE``. Path entries
   ending in ``/bin`` or ``/sbin`` are automatically converted to their
   parent directories:

   * ``PATH``

6. Search paths stored in the CMake :ref:`User Package Registry`.
   This can be skipped if ``NO_CMAKE_PACKAGE_REGISTRY`` is passed or by
   setting the variable :variable:`CMAKE_FIND_USE_PACKAGE_REGISTRY`
   to ``FALSE`` or the deprecated variable
   :variable:`CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY` to ``TRUE``.

   See the :manual:`cmake-packages(7)` manual for details on the user
   package registry.

7. Search CMake variables defined in the Platform files for the
   current system. The searching of :variable:`CMAKE_INSTALL_PREFIX` and
   :variable:`CMAKE_STAGING_PREFIX` can be
   skipped if ``NO_CMAKE_INSTALL_PREFIX`` is passed or by setting the
   :variable:`CMAKE_FIND_USE_INSTALL_PREFIX` to ``FALSE``. All these locations
   can be skipped if ``NO_CMAKE_SYSTEM_PATH`` is passed or by setting the
   :variable:`CMAKE_FIND_USE_CMAKE_SYSTEM_PATH` to ``FALSE``:

   * :variable:`CMAKE_SYSTEM_PREFIX_PATH`
   * :variable:`CMAKE_SYSTEM_FRAMEWORK_PATH`
   * :variable:`CMAKE_SYSTEM_APPBUNDLE_PATH`

   The platform paths that these variables contain are locations that
   typically include installed software. An example being ``/usr/local`` for
   UNIX based platforms.

8. Search paths stored in the CMake :ref:`System Package Registry`.
   This can be skipped if ``NO_CMAKE_SYSTEM_PACKAGE_REGISTRY`` is passed
   or by setting the :variable:`CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY`
   variable to ``FALSE`` or the deprecated variable
   :variable:`CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY` to ``TRUE``.

   See the :manual:`cmake-packages(7)` manual for details on the system
   package registry.

9. Search paths specified by the ``PATHS`` option.  These are typically
   hard-coded guesses.

The :variable:`CMAKE_IGNORE_PATH`, :variable:`CMAKE_IGNORE_PREFIX_PATH`,
:variable:`CMAKE_SYSTEM_IGNORE_PATH` and
:variable:`CMAKE_SYSTEM_IGNORE_PREFIX_PATH` variables can also cause some
of the above locations to be ignored.

Paths are searched in the order described above.  The first viable package
configuration file found is used, even if a newer version of the package
resides later in the list of search paths.

For search paths which contain glob expressions (``*``), directories matching
the glob are searched in natural, descending order by default. This behavior
can be overridden by setting variables :variable:`CMAKE_FIND_PACKAGE_SORT_ORDER`
and :variable:`CMAKE_FIND_PACKAGE_SORT_DIRECTION` accordingly. Those variables
determine the order in which CMake considers glob matches. For example, if the
file system contains the package configuration files

::

  <prefix>/example-1.2/example-config.cmake
  <prefix>/example-1.10/example-config.cmake
  <prefix>/share/example-2.0/example-config.cmake

then ``find_package(example)`` will (when the aforementioned variables are
unset) pick ``example-1.10`` (assuming both ``example-1.2`` and ``example-1.10``
are viable). Note however that ``find_package`` will *not* find ``example-2.0``,
because one of the other two will be found first.

To control the order in which ``find_package`` searches directories that match
a glob expression, use :variable:`CMAKE_FIND_PACKAGE_SORT_ORDER` and
:variable:`CMAKE_FIND_PACKAGE_SORT_DIRECTION`.
For instance, to cause the above example to select ``example-1.2``,
one can set

.. code-block:: cmake

  set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
  set(CMAKE_FIND_PACKAGE_SORT_DIRECTION ASC)

before calling ``find_package``.

.. versionadded:: 3.16
   Added the ``CMAKE_FIND_USE_<CATEGORY>`` variables to globally disable
   various search locations.

.. versionchanged:: 4.0
   The variables :variable:`CMAKE_FIND_PACKAGE_SORT_ORDER` and
   :variable:`CMAKE_FIND_PACKAGE_SORT_DIRECTION` now also control the order
   in which ``find_package`` searches directories matching the glob expression
   in the search paths ``<prefix>/<name>.framework/Versions/*/Resources/``
   and ``<prefix>/<name>.framework/Versions/*/Resources/CMake``.  In previous
   versions of CMake, this order was unspecified.

.. versionchanged:: 4.2
   When encountering multiple viable matches, ``find_package`` now picks the
   one with the most recent version by default. In previous versions of CMake,
   the result was unspecified. Accordingly, the default of
   :variable:`CMAKE_FIND_PACKAGE_SORT_ORDER` has changed from ``NONE`` to
   ``NATURAL`` and :variable:`CMAKE_FIND_PACKAGE_SORT_DIRECTION`
   now defaults to ``DEC`` (descending) instead of ``ASC`` (ascending).


.. include:: include/FIND_XXX_ROOT.rst
.. include:: include/FIND_XXX_ORDER.rst

By default the value stored in the result variable will be the path at
which the file is found.  The :variable:`CMAKE_FIND_PACKAGE_RESOLVE_SYMLINKS`
variable may be set to ``TRUE`` before calling ``find_package`` in order
to resolve symbolic links and store the real path to the file.

Every non-REQUIRED ``find_package`` call can be disabled or made REQUIRED:

* Setting the :variable:`CMAKE_DISABLE_FIND_PACKAGE_<PackageName>` variable
  to ``TRUE`` disables the package.  This also disables redirection to a
  package provided by :module:`FetchContent`.

* Setting the :variable:`CMAKE_REQUIRE_FIND_PACKAGE_<PackageName>` variable
  to ``TRUE`` makes the package REQUIRED.

Setting both variables to ``TRUE`` simultaneously is an error.

The :variable:`CMAKE_REQUIRE_FIND_PACKAGE_<PackageName>` variable takes priority
over the ``OPTIONAL`` keyword in determining whether a package is required.

.. _`version selection`:

Config Mode Version Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
  When Config mode is used, this version selection process is applied
  regardless of whether the :ref:`full <full signature>` or
  :ref:`basic <basic signature>` signature was given.

When the ``[version]`` argument is given, Config mode will only find a
version of the package that claims compatibility with the requested
version (see :ref:`format specification <FIND_PACKAGE_VERSION_FORMAT>`).  If
the ``EXACT`` option is given, only a version of the package claiming an exact
match of the requested version may be found.  CMake does not establish any
convention for the meaning of version numbers.

.. _`cmake script version selection`:

CMake-script
""""""""""""

For CMake-script package configuration files, package version numbers are
checked by "version" files provided by the packages themselves or by
:module:`FetchContent`.  For a candidate package configuration file
``<config-file>.cmake`` the corresponding version file is located next
to it and named either ``<config-file>-version.cmake`` or
``<config-file>Version.cmake``.  If no such version file is available
then the configuration file is assumed to not be compatible with any
requested version.  A basic version file containing generic version
matching code can be created using the
:module:`CMakePackageConfigHelpers` module.  When a version file
is found it is loaded to check the requested version number.  The
version file is loaded in a nested scope in which the following
variables have been defined:

``PACKAGE_FIND_NAME``
  The ``<PackageName>``
``PACKAGE_FIND_VERSION``
  Full requested version string
``PACKAGE_FIND_VERSION_MAJOR``
  Major version if requested, else 0
``PACKAGE_FIND_VERSION_MINOR``
  Minor version if requested, else 0
``PACKAGE_FIND_VERSION_PATCH``
  Patch version if requested, else 0
``PACKAGE_FIND_VERSION_TWEAK``
  Tweak version if requested, else 0
``PACKAGE_FIND_VERSION_COUNT``
  Number of version components, 0 to 4

When a version range is specified, the above version variables will hold
values based on the lower end of the version range.  This is to preserve
compatibility with packages that have not been implemented to expect version
ranges.  In addition, the version range will be described by the following
variables:

``PACKAGE_FIND_VERSION_RANGE``
  Full requested version range string
``PACKAGE_FIND_VERSION_RANGE_MIN``
  This specifies whether the lower end point of the version range should be
  included or excluded.  Currently, the only supported value for this variable
  is ``INCLUDE``.
``PACKAGE_FIND_VERSION_RANGE_MAX``
  This specifies whether the upper end point of the version range should be
  included or excluded.  The supported values for this variable are
  ``INCLUDE`` and ``EXCLUDE``.

``PACKAGE_FIND_VERSION_MIN``
  Full requested version string of the lower end point of the range
``PACKAGE_FIND_VERSION_MIN_MAJOR``
  Major version of the lower end point if requested, else 0
``PACKAGE_FIND_VERSION_MIN_MINOR``
  Minor version of the lower end point if requested, else 0
``PACKAGE_FIND_VERSION_MIN_PATCH``
  Patch version of the lower end point if requested, else 0
``PACKAGE_FIND_VERSION_MIN_TWEAK``
  Tweak version of the lower end point if requested, else 0
``PACKAGE_FIND_VERSION_MIN_COUNT``
  Number of version components of the lower end point, 0 to 4

``PACKAGE_FIND_VERSION_MAX``
  Full requested version string of the upper end point of the range
``PACKAGE_FIND_VERSION_MAX_MAJOR``
  Major version of the upper end point if requested, else 0
``PACKAGE_FIND_VERSION_MAX_MINOR``
  Minor version of the upper end point if requested, else 0
``PACKAGE_FIND_VERSION_MAX_PATCH``
  Patch version of the upper end point if requested, else 0
``PACKAGE_FIND_VERSION_MAX_TWEAK``
  Tweak version of the upper end point if requested, else 0
``PACKAGE_FIND_VERSION_MAX_COUNT``
  Number of version components of the upper end point, 0 to 4

Regardless of whether a single version or a version range is specified, the
variable ``PACKAGE_FIND_VERSION_COMPLETE`` will be defined and will hold
the full requested version string as specified.

The version file checks whether it satisfies the requested version and
sets these variables:

``PACKAGE_VERSION``
  Full provided version string
``PACKAGE_VERSION_EXACT``
  True if version is exact match
``PACKAGE_VERSION_COMPATIBLE``
  True if version is compatible
``PACKAGE_VERSION_UNSUITABLE``
  True if unsuitable as any version

These variables are checked by the ``find_package`` command to determine
whether the configuration file provides an acceptable version.  They
are not available after the ``find_package`` call returns.  If the version
is acceptable, the following variables are set:

``<PackageName>_VERSION``
  Full provided version string
``<PackageName>_VERSION_MAJOR``
  Major version if provided, else 0
``<PackageName>_VERSION_MINOR``
  Minor version if provided, else 0
``<PackageName>_VERSION_PATCH``
  Patch version if provided, else 0
``<PackageName>_VERSION_TWEAK``
  Tweak version if provided, else 0
``<PackageName>_VERSION_COUNT``
  Number of version components, 0 to 4

and the corresponding package configuration file is loaded.

.. note::
  While the exact behavior of version matching is determined by the individual
  package, many packages use :command:`write_basic_package_version_file` to
  supply this logic.  The version check scripts this produces have some notable
  caveats with respect to version ranges:

  * The upper end of a version range acts as a hard limit on what versions will
    be accepted.  Thus, while a request for version ``1.4.0`` might be
    satisfied by a package whose version is ``1.6.0`` and which advertises
    'same major version' compatibility, the same package will be rejected if
    the requested version range is ``1.4.0...1.5.0``.

  * Both ends of the version range must match the package's advertised
    compatibility level. For example, if a package advertises 'same major and
    minor version' compatibility, requesting the version range
    ``1.4.0...<1.5.5`` or ``1.4.0...1.5.0`` will result in that package being
    rejected, even if the package version is ``1.4.1``.

  As a result, it is not possible to use a version range to extend the range
  of compatible package versions that will be accepted.

.. _`cps version selection`:

|CPS|
"""""

For |CPS| package configuration files, package version numbers are checked by
CMake according to the set of recognized version schemas. At present, the
following schemas are recognized:

  ``simple``
    Version numbers are a tuple of integers followed by an optional trailing
    segment which is ignored with respect to version comparisons.

  ``custom``
    The mechanism for interpreting version numbers is unspecified.  The version
    strings must match exactly for the package to be accepted.

Refer to |cps-version_schema|_ for a more detailed explanation of each schema
and how comparisons for each are performed.  Note that the specification may
include schemas that are not supported by CMake.

In addition to the package's ``version``, CPS allows packages to optionally
specify a |cps-compat_version|_, which is the oldest version for which the
package provides compatibility.  That is, the package warrants that a consumer
expecting the ``compat_version`` should be able to use the package, even if the
package's actual version is newer.  If not specified, the ``compat_version``
is implicitly equal to the package version, i.e. no backwards compatibility is
provided.

.. TODO Rework the preceding paragraph when COMPAT_VERSION has broader support
        in CMake.

When a package uses a recognized schema, CMake will determine the package's
acceptability according to the following rules:

* If ``EXACT`` was specified, or if the package does not supply a
  ``compat_version``, the package's ``version`` must equal the requested
  version.

* Otherwise:

  * The package's ``version`` must be greater than or equal to the requested
    (minimum) version, and

  * the package's ``compat_version`` must be less than or equal to the
    requested (minimum) version, and

  * if a requested maximum version was given, it must be greater than (or equal
    to, depending on whether the maximum version is specified as inclusive or
    exclusive) the package's ``version``.

.. note::
  This implementation of range matching was chosen in order to most closely
  match the behavior of :command:`write_basic_package_version_file`, albeit
  without the case where an overly broad range matches nothing.

For packages using the ``simple`` version schema, if the version is acceptable,
the following variables are set:

``<PackageName>_VERSION``
  Full provided version string
``<PackageName>_VERSION_MAJOR``
  Major version if provided, else 0
``<PackageName>_VERSION_MINOR``
  Minor version if provided, else 0
``<PackageName>_VERSION_PATCH``
  Patch version if provided, else 0
``<PackageName>_VERSION_TWEAK``
  Tweak version if provided, else 0
``<PackageName>_VERSION_COUNT``
  Number of version components, non-negative

Package File Interface Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When loading a find module or CMake-script package configuration file,
``find_package`` defines variables to provide information about the call
arguments (and restores their original state before returning):

``CMAKE_FIND_PACKAGE_NAME``
  The ``<PackageName>`` which is searched for
``<PackageName>_FIND_REQUIRED``
  True if ``REQUIRED`` option was given
``<PackageName>_FIND_QUIETLY``
  True if ``QUIET`` option was given
``<PackageName>_FIND_REGISTRY_VIEW``
  The requested view if ``REGISTRY_VIEW`` option was given
``<PackageName>_FIND_VERSION``
  Full requested version string
``<PackageName>_FIND_VERSION_MAJOR``
  Major version if requested, else 0
``<PackageName>_FIND_VERSION_MINOR``
  Minor version if requested, else 0
``<PackageName>_FIND_VERSION_PATCH``
  Patch version if requested, else 0
``<PackageName>_FIND_VERSION_TWEAK``
  Tweak version if requested, else 0
``<PackageName>_FIND_VERSION_COUNT``
  Number of version components, 0 to 4
``<PackageName>_FIND_VERSION_EXACT``
  True if ``EXACT`` option was given
``<PackageName>_FIND_COMPONENTS``
  List of specified components (required and optional)
``<PackageName>_FIND_REQUIRED_<c>``
  True if component ``<c>`` is required,
  false if component ``<c>`` is optional

When a version range is specified, the above version variables will hold
values based on the lower end of the version range.  This is to preserve
compatibility with packages that have not been implemented to expect version
ranges.  In addition, the version range will be described by the following
variables:

``<PackageName>_FIND_VERSION_RANGE``
  Full requested version range string
``<PackageName>_FIND_VERSION_RANGE_MIN``
  This specifies whether the lower end point of the version range is
  included or excluded.  Currently, ``INCLUDE`` is the only supported value.
``<PackageName>_FIND_VERSION_RANGE_MAX``
  This specifies whether the upper end point of the version range is
  included or excluded.  The possible values for this variable are
  ``INCLUDE`` or ``EXCLUDE``.

``<PackageName>_FIND_VERSION_MIN``
  Full requested version string of the lower end point of the range
``<PackageName>_FIND_VERSION_MIN_MAJOR``
  Major version of the lower end point if requested, else 0
``<PackageName>_FIND_VERSION_MIN_MINOR``
  Minor version of the lower end point if requested, else 0
``<PackageName>_FIND_VERSION_MIN_PATCH``
  Patch version of the lower end point if requested, else 0
``<PackageName>_FIND_VERSION_MIN_TWEAK``
  Tweak version of the lower end point if requested, else 0
``<PackageName>_FIND_VERSION_MIN_COUNT``
  Number of version components of the lower end point, 0 to 4

``<PackageName>_FIND_VERSION_MAX``
  Full requested version string of the upper end point of the range
``<PackageName>_FIND_VERSION_MAX_MAJOR``
  Major version of the upper end point if requested, else 0
``<PackageName>_FIND_VERSION_MAX_MINOR``
  Minor version of the upper end point if requested, else 0
``<PackageName>_FIND_VERSION_MAX_PATCH``
  Patch version of the upper end point if requested, else 0
``<PackageName>_FIND_VERSION_MAX_TWEAK``
  Tweak version of the upper end point if requested, else 0
``<PackageName>_FIND_VERSION_MAX_COUNT``
  Number of version components of the upper end point, 0 to 4

Regardless of whether a single version or a version range is specified, the
variable ``<PackageName>_FIND_VERSION_COMPLETE`` will be defined and will hold
the full requested version string as specified.

In Module mode the loaded find module is responsible to honor the
request detailed by these variables; see the find module for details.
In Config mode ``find_package`` handles ``REQUIRED``, ``QUIET``, and
``[version]`` options automatically but leaves it to the package
configuration file to handle components in a way that makes sense
for the package.  The package configuration file may set
``<PackageName>_FOUND`` to false to tell ``find_package`` that component
requirements are not satisfied.

.. _CPS: https://cps-org.github.io/cps/
.. |CPS| replace:: Common Package Specification

.. _cps-compat_version: https://cps-org.github.io/cps/schema.html#compat-version
.. |cps-compat_version| replace:: ``compat_version``

.. _cps-version_schema: https://cps-org.github.io/cps/schema.html#version-schema
.. |cps-version_schema| replace:: ``version_schema``

CPS Transitive Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A |CPS| package description consists of one or more components which may in
turn depend on other components either internal or external to the package.
When external components are required, the providing package is noted as
a package-level requirement of the package.  Additionally, the set of required
components is typically noted in said external package requirement.

Where a CMake-script package description would use the
:command:`find_dependency` command to handle transitive dependencies, CMake
handles transitive dependencies for CPS itself using an internally nested
``find_package`` call.  This call can resolve CPS package dependencies via
*either* another CPS package, or via a CMake-script package.  The manner in
which the CPS component dependencies are handled is subject to some caveats.

When the candidate for resolving a transitive dependency is another CPS
package, things are simple; ``COMPONENTS`` and CPS "components" are directly
comparable (and are effectively synonymous with CMake "imported targets").
CMake-script packages, however, are encouraged to (and often do) check that
required components were found, whether or not the package describes separate
components.  Additionally, even those that do describe components typically do
not have the same correlation to imported targets that is normal for CPS.  As
a result, passing the set of required components declared by a CPS package to
``COMPONENTS`` would result in spurious failures to resolve dependencies.

To address this, if a candidate for resolving a CPS transitive dependency is a
CMake-script package, CMake passes the required components as declared by the
consuming CPS package as ``OPTIONAL_COMPONENTS`` and performs a separate,
internal check that the candidate package supplied the required imported
targets.  Those targets must be named ``<PackageName>::<ComponentName>``, in
conformance with CPS convention, or the check will consider the package not
found.
