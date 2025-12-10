CMake 3.10 Release Notes
************************

.. only:: html

  .. contents::

Changes made since CMake 3.9 include the following.

New Features
============

Platforms
---------

* The `flang`_ Fortran compiler is now supported, with compiler id ``Flang``.

* A new minimal platform file for ``Midipix`` was added.

* Support for the MSVC ARM64 architecture was added.
  Visual Studio 2017 Update 4 and above offer an ARM64 toolchain.

* Support for the IAR ARM Compiler was improved.

.. _`flang`: https://github.com/flang-compiler/flang

Generators
----------

* The :ref:`Makefile Generators` and the :generator:`Ninja` generator learned
  to add compiler launcher tools like ccache along with the compiler for the
  ``CUDA`` language (``C`` and ``CXX`` were supported previously).  See the
  :variable:`CMAKE_<LANG>_COMPILER_LAUNCHER` variable and
  :prop_tgt:`<LANG>_COMPILER_LAUNCHER` target property for details.

* The :generator:`CodeBlocks` extra generator learned to optionally exclude
  files from outside the project root directory from the generated project.
  See the :variable:`CMAKE_CODEBLOCKS_EXCLUDE_EXTERNAL_FILES` variable.

Commands
--------

* The :command:`cmake_host_system_information` command learned more keys
  to get information about the processor capabilities and the host OS
  version.

* The :command:`configure_file` command learned to support indented
  ``#  cmakedefine`` and ``#  cmakedefine01``. Spaces and/or tabs between
  the ``#`` character and the ``cmakedefine``/``cmakedefine01`` words
  are now understood and preserved in the output.

* The :command:`execute_process` command gained a ``RESULTS_VARIABLE``
  option to collect a list of results from all children in a pipeline
  of processes when multiple ``COMMAND`` arguments are given.

* The :command:`include_guard` command was introduced to allow guarding
  CMake scripts from being included more than once. The command supports
  ``DIRECTORY`` and ``GLOBAL`` options to adjust the corresponding include guard
  scope. If no options given, include guard is similar to basic variable-based
  check.

* The :command:`string` command learned a new ``PREPEND`` subcommand.

* The :command:`string(TIMESTAMP)` command now supports ``%A``
  for full weekday name and ``%B`` for full month name.

Variables
---------

* A :variable:`CMAKE_DIRECTORY_LABELS` variable was added to specify
  labels for all tests in a directory.

Properties
----------

* A :prop_tgt:`<LANG>_CPPCHECK` target property and supporting
  :variable:`CMAKE_<LANG>_CPPCHECK` variable were introduced to tell
  the :ref:`Makefile Generators` and the :generator:`Ninja` generator to
  run ``cppcheck`` with the compiler for ``C`` and ``CXX`` languages.

* A :prop_dir:`LABELS` directory property was added to specify labels
  for all targets and tests in a directory.

* A :prop_dir:`TEST_INCLUDE_FILES` directory property was added to
  list any number of files to be included when running tests with
  :manual:`ctest(1)`.  This generalizes the :prop_dir:`TEST_INCLUDE_FILE`
  property.

* The :prop_tgt:`VS_DOTNET_REFERENCEPROP_<refname>_TAG_<tagname>`
  target property was added to support custom XML tags for reference
  assemblies in C# targets.

* Source file properties :prop_sf:`VS_SHADER_OUTPUT_HEADER_FILE` and
  :prop_sf:`VS_SHADER_VARIABLE_NAME` have been added to specify more
  details of ``.hlsl`` sources with :ref:`Visual Studio Generators`.

Modules
-------

* The :module:`FindCurses` module gained a ``CURSES_NEED_WIDE`` option
  to request the wide-character variant.

* The :module:`FindEXPAT` module now provides imported targets.

* The :module:`FindFreetype` module now provides imported targets.

* :module:`FindMPI` gained a number of new features, including:

  * Language-specific components have been added to the module.
  * Many more MPI environments are now supported.
  * The environmental support for Fortran has been improved.
  * A user now has fine-grained control over the MPI selection process,
    including passing custom parameters to the MPI compiler.
  * The version of the implemented MPI standard is now being exposed.
  * MPI-2 C++ bindings can now be detected and also suppressed if so desired.
  * The available Fortran bindings are now being detected and verified.
  * Various MPI-3 information can be requested, including the library version
    and Fortran capabilities of the individual bindings.
  * Statically linked MPI implementations are supported.

* A :module:`FindOpenACC` module was added to detect compiler support
  for OpenACC.  Currently only supports PGI, GNU and Cray compilers.

* The :module:`FindOpenGL` module gained support for GLVND on Linux.

* The :module:`FindOpenMP` module gained support for
  language-specific components.

* A :module:`FindPatch` module was added to find the ``patch``
  command-line executable.

* The :module:`FindProtobuf` module :command:`protobuf_generate_cpp` command
  gained a ``DESCRIPTORS`` option to generate descriptor files.

* The :module:`GoogleTest` module gained a new command
  :command:`gtest_discover_tests` implementing dynamic (build-time) test
  discovery.  Unlike the source parsing approach, dynamic discovery executes
  the test (in 'list available tests' mode) at build time to discover tests.
  This is robust against unusual ways of labeling tests, provides much better
  support for advanced features such as parameterized tests, and does not
  require re-running CMake to discover added or removed tests within a test
  executable.  Note that a breaking change was made in CMake 3.10.3 to address
  an ambiguity of the ``TIMEOUT`` keyword (see :ref:`Release Notes 3.10.3`).

* The :module:`InstallRequiredSystemLibraries` module gained support
  for installing Intel compiler runtimes.

Autogen
-------

* When using :prop_tgt:`AUTOMOC` or :prop_tgt:`AUTOUIC` with a
  multi configuration generator (e.g. :generator:`Xcode`),
  included ``*.moc``,  ``moc_*.cpp`` and ``ui_*.h`` files are generated in
  ``<AUTOGEN_BUILD_DIR>/include_<CONFIG>`` instead of
  ``<AUTOGEN_BUILD_DIR>/include``.

* When using :prop_tgt:`AUTOMOC` or :prop_tgt:`AUTOUIC`,
  source files that are :prop_sf:`GENERATED` will be processed as well.
  They were ignored by :prop_tgt:`AUTOMOC` and :prop_tgt:`AUTOUIC`
  in earlier releases.
  See policy :policy:`CMP0071`.

* When using :prop_tgt:`AUTOMOC`, CMake searches for the strings ``Q_OBJECT``,
  ``Q_GADGET`` or ``Q_NAMESPACE`` in a source file to determine if it needs
  to be ``moc`` processed.  The new :variable:`CMAKE_AUTOMOC_MACRO_NAMES`
  variable and :prop_tgt:`AUTOMOC_MACRO_NAMES` target property may be set
  to register additional strings (macro names) to search for.

* When using :prop_tgt:`AUTOMOC`, the new
  :variable:`CMAKE_AUTOMOC_COMPILER_PREDEFINES` variable and
  :prop_tgt:`AUTOMOC_COMPILER_PREDEFINES` target property specify whether
  to enable or disable the generation of the compiler pre definitions file
  ``moc_predefs.h``.

CTest
-----

* A :variable:`CTEST_LABELS_FOR_SUBPROJECTS` CTest module variable and CTest
  script variable were added to specify a list of labels that should be
  treated as subprojects by CDash. To use this value in both the CTest module
  and the ctest command line :ref:`Dashboard Client` mode (e.g. ``ctest -S``)
  set it in the ``CTestConfig.cmake`` config file.

CPack
-----

* A :cpack_gen:`CPack FreeBSD Generator` was added for FreeBSD ``pkg(8)``.

* The :cpack_gen:`CPack DEB Generator` was enabled on Windows.  While not
  fully featured (due to the lack of external UNIX tools) this will allow
  building basic cross-platform Debian packages.

* The :cpack_gen:`CPack DEB Generator` learned to set package release
  version in ``Version`` info property.
  See the :variable:`CPACK_DEBIAN_PACKAGE_RELEASE` variable.

* The :cpack_gen:`CPack DEB Generator` learned more strict package
  version checking that complies with Debian rules.

* The :module:`CPackIFW` module :command:`cpack_ifw_configure_component` and
  :command:`cpack_ifw_configure_component_group` commands gained a new
  ``REPLACES`` and ``CHECKABLE`` options.

* The :cpack_gen:`CPack IFW Generator` gained new
  :variable:`CPACK_IFW_PACKAGE_FILE_EXTENSION` variable to customize
  target binary format.

* The :cpack_gen:`CPack IFW Generator` gained new
  :variable:`CPACK_IFW_REPOSITORIES_DIRECTORIES` variable to specify
  additional repositories dirs that will be used to resolve and
  repack dependent components. This feature is only available when
  using QtIFW 3.1 or later.

* The :cpack_gen:`CPack RPM Generator` and :cpack_gen:`CPack DEB Generator`
  learned to set the package epoch version.
  See :variable:`CPACK_RPM_PACKAGE_EPOCH` and
  :variable:`CPACK_DEBIAN_PACKAGE_EPOCH` variables.

Other
-----

* The :manual:`cmake(1)` ``-E`` mode gained support for ``sha1sum``,
  ``sha224sum``, ``sha256sum``, ``sha384sum``, and ``sha512sum``.

* The graphviz output now distinguishes among the different dependency types
  ``PUBLIC``, ``PRIVATE`` and ``INTERFACE`` and represents them in the output
  graph as solid, dashed and dotted edges.

Deprecated and Removed Features
===============================

* Support for building CMake itself with C++98 compilers was dropped.
  CMake is now implemented using C++11.

* Support for building CMake on HP-UX has been dropped pending better
  support for C++11 and a port of libuv.  See `CMake Issue 17137`_.
  Use CMake 3.9 or lower instead for HP-UX support.

.. _`CMake Issue 17137`: https://gitlab.kitware.com/cmake/cmake/-/issues/17137

Other Changes
=============

* On FreeBSD the C++ compiler named ``c++`` is now the preferred default.

* The :command:`file(GENERATE)` command now interprets relative paths
  given to its ``OUTPUT`` and ``INPUT`` arguments with respect to the
  caller's current binary and source directories, respectively.
  See policy :policy:`CMP0070`.

* The :command:`get_filename_component` ``PROGRAM`` mode semantics
  have been revised to not tolerate unquoted spaces in the path
  to the program while also accepting arguments.  While technically
  incompatible with the old behavior, it is expected that behavior
  under typical use cases with properly-quoted command-lines has
  not changed.

Updates
=======

Changes made since CMake 3.10.0 include the following.

3.10.1
------

* The :manual:`cmake-server(7)` ``codemodel`` response cross-references
  field added by 3.10.0 has been dropped due to excessive memory usage.
  Another approach will be needed to provide backtrace information.

.. _`Release Notes 3.10.3`:

3.10.3
------

* CMake 3.10.1 added a ``TIMEOUT`` option to :command:`gtest_discover_tests`
  from the :module:`GoogleTest` module.  That keyword clashed with the
  ``TIMEOUT`` test property, which is one of the common properties that
  would be set with the command's ``PROPERTIES`` keyword, usually leading
  to legal but unintended behavior.  The keyword was changed to
  ``DISCOVERY_TIMEOUT`` in CMake 3.10.3 to address this problem.  The
  ambiguous behavior of the :command:`gtest_discover_tests` command's
  ``TIMEOUT`` keyword in 3.10.1 and 3.10.2 has not been preserved.
