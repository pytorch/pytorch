.. cmake-manual-description: CMake Command-Line Reference

cmake(1)
********

Synopsis
========

.. parsed-literal::

 `Generate a Project Buildsystem`_
  cmake [<options>] -B <path-to-build> [-S <path-to-source>]
  cmake [<options>] <path-to-source | path-to-existing-build>

 `Build a Project`_
  cmake --build <dir> [<options>] [-- <build-tool-options>]

 `Install a Project`_
  cmake --install <dir> [<options>]

 `Open a Project`_
  cmake --open <dir>

 `Run a Script`_
  cmake [-D <var>=<value>]... -P <cmake-script-file>

 `Run a Command-Line Tool`_
  cmake -E <command> [<options>]

 `Run the Find-Package Tool`_
  cmake --find-package [<options>]

 `Run a Workflow Preset`_
  cmake --workflow <options>

 `View Help`_
  cmake --help[-<topic>]

Description
===========

The :program:`cmake` executable is the command-line interface of the cross-platform
buildsystem generator CMake.  The above `Synopsis`_ lists various actions
the tool can perform as described in sections below.

To build a software project with CMake, `Generate a Project Buildsystem`_.
Optionally use :program:`cmake` to `Build a Project`_, `Install a Project`_ or just
run the corresponding build tool (e.g. ``make``) directly.  :program:`cmake` can also
be used to `View Help`_.

The other actions are meant for use by software developers writing
scripts in the :manual:`CMake language <cmake-language(7)>` to support
their builds.

For graphical user interfaces that may be used in place of :program:`cmake`,
see :manual:`ccmake <ccmake(1)>` and :manual:`cmake-gui <cmake-gui(1)>`.
For command-line interfaces to the CMake testing and packaging facilities,
see :manual:`ctest <ctest(1)>` and :manual:`cpack <cpack(1)>`.

For more information on CMake at large, `see also`_ the links at the end
of this manual.


Introduction to CMake Buildsystems
==================================

A *buildsystem* describes how to build a project's executables and libraries
from its source code using a *build tool* to automate the process.  For
example, a buildsystem may be a ``Makefile`` for use with a command-line
``make`` tool or a project file for an Integrated Development Environment
(IDE).  In order to avoid maintaining multiple such buildsystems, a project
may specify its buildsystem abstractly using files written in the
:manual:`CMake language <cmake-language(7)>`.  From these files CMake
generates a preferred buildsystem locally for each user through a backend
called a *generator*.

To generate a buildsystem with CMake, the following must be selected:

Source Tree
  The top-level directory containing source files provided by the project.
  The project specifies its buildsystem using files as described in the
  :manual:`cmake-language(7)` manual, starting with a top-level file named
  ``CMakeLists.txt``.  These files specify build targets and their
  dependencies as described in the :manual:`cmake-buildsystem(7)` manual.

Build Tree
  The top-level directory in which buildsystem files and build output
  artifacts (e.g. executables and libraries) are to be stored.
  CMake will write a ``CMakeCache.txt`` file to identify the directory
  as a build tree and store persistent information such as buildsystem
  configuration options.

  To maintain a pristine source tree, perform an *out-of-source* build
  by using a separate dedicated build tree.  An *in-source* build in
  which the build tree is placed in the same directory as the source
  tree is also supported, but discouraged.

Generator
  This chooses the kind of buildsystem to generate.  See the
  :manual:`cmake-generators(7)` manual for documentation of all generators.
  Run :option:`cmake --help` to see a list of generators available locally.
  Optionally use the :option:`-G <cmake -G>` option below to specify a
  generator, or simply accept the default CMake chooses for the current
  platform.

  When using one of the :ref:`Command-Line Build Tool Generators`
  CMake expects that the environment needed by the compiler toolchain
  is already configured in the shell.  When using one of the
  :ref:`IDE Build Tool Generators`, no particular environment is needed.

.. _`Generate a Project Buildsystem`:

Generate a Project Buildsystem
==============================

Run CMake with one of the following command signatures to specify the
source and build trees and generate a buildsystem:

``cmake [<options>] -B <path-to-build> [-S <path-to-source>]``

  .. versionadded:: 3.13

  Uses ``<path-to-build>`` as the build tree and ``<path-to-source>``
  as the source tree.  The specified paths may be absolute or relative
  to the current working directory.  The source tree must contain a
  ``CMakeLists.txt`` file.  The build tree will be created automatically
  if it does not already exist.  For example:

  .. code-block:: console

    $ cmake -S src -B build

``cmake [<options>] <path-to-source>``
  Uses the current working directory as the build tree, and
  ``<path-to-source>`` as the source tree.  The specified path may
  be absolute or relative to the current working directory.
  The source tree must contain a ``CMakeLists.txt`` file and must
  *not* contain a ``CMakeCache.txt`` file because the latter
  identifies an existing build tree.  For example:

  .. code-block:: console

    $ mkdir build ; cd build
    $ cmake ../src

``cmake [<options>] <path-to-existing-build>``
  Uses ``<path-to-existing-build>`` as the build tree, and loads the
  path to the source tree from its ``CMakeCache.txt`` file, which must
  have already been generated by a previous run of CMake.  The specified
  path may be absolute or relative to the current working directory.
  For example:

  .. code-block:: console

    $ cd build
    $ cmake .

In all cases the ``<options>`` may be zero or more of the `Options`_ below.

The above styles for specifying the source and build trees may be mixed.
Paths specified with :option:`-S <cmake -S>` or :option:`-B <cmake -B>`
are always classified as source or build trees, respectively.  Paths
specified with plain arguments are classified based on their content
and the types of paths given earlier.  If only one type of path is given,
the current working directory (cwd) is used for the other.  For example:

============================== ============ ===========
 Command Line                   Source Dir   Build Dir
============================== ============ ===========
 ``cmake -B build``             *cwd*        ``build``
 ``cmake -B build src``         ``src``      ``build``
 ``cmake -B build -S src``      ``src``      ``build``
 ``cmake src``                  ``src``      *cwd*
 ``cmake build`` (existing)     *loaded*     ``build``
 ``cmake -S src``               ``src``      *cwd*
 ``cmake -S src build``         ``src``      ``build``
 ``cmake -S src -B build``      ``src``      ``build``
============================== ============ ===========

.. versionchanged:: 3.23

  CMake warns when multiple source paths are specified.  This has never
  been officially documented or supported, but older versions accidentally
  accepted multiple source paths and used the last path specified.
  Avoid passing multiple source path arguments.

After generating a buildsystem one may use the corresponding native
build tool to build the project.  For example, after using the
:generator:`Unix Makefiles` generator one may run ``make`` directly:

  .. code-block:: console

    $ make
    $ make install

Alternatively, one may use :program:`cmake` to `Build a Project`_ by
automatically choosing and invoking the appropriate native build tool.

.. _`CMake Options`:

Options
-------

.. program:: cmake

.. include:: include/OPTIONS_BUILD.rst

.. option:: --fresh

 .. versionadded:: 3.24

 Perform a fresh configuration of the build tree.
 This removes any existing ``CMakeCache.txt`` file and associated
 ``CMakeFiles/`` directory, and recreates them from scratch.

 .. versionchanged:: 3.30

   For dependencies previously populated by :module:`FetchContent` with the
   ``NEW`` setting for policy :policy:`CMP0168`, their stamp and script files
   from any previous run will be removed. The download, update, and patch
   steps will therefore be forced to re-execute.

.. option:: -L[A][H]

 List non-advanced cached variables.

 List ``CACHE`` variables will run CMake and list all the variables from
 the CMake ``CACHE`` that are not marked as ``INTERNAL`` or :prop_cache:`ADVANCED`.
 This will effectively display current CMake settings, which can then be
 changed with :option:`-D <cmake -D>` option.  Changing some of the variables
 may result in more variables being created.  If ``A`` is specified, then it
 will display also advanced variables.  If ``H`` is specified, it will also
 display help for each variable.

.. option:: -LR[A][H] <regex>

 .. versionadded:: 3.31

 Show specific non-advanced cached variables

 Show non-``INTERNAL`` nor :prop_cache:`ADVANCED` variables from the CMake
 ``CACHE`` that match the given regex. If ``A`` is specified, then it
 will also show advanced variables.  If ``H`` is specified, it will also
 display help for each variable.

.. option:: -N

 View mode only.

 Only load the cache.  Do not actually run configure and generate
 steps.

.. option:: --graphviz=<file>

  Generate `Graphviz <https://www.graphviz.org/>`_ of dependencies

  This option generates a graphviz input file that will contain all the
  library and executable dependencies in the project showing the
  dependencies between the targets in a project, as well as external libraries
  which are linked against.

  When running CMake with the ``--graphviz=foo.dot`` option, it produces:

  * a ``foo.dot`` file, showing all dependencies in the project
  * a ``foo.dot.<target>`` file for each target, showing on which other targets
    it depends
  * a ``foo.dot.<target>.dependers`` file for each target, showing which other
    targets depend on it

  Those .dot files can be converted to images using the *dot* command from the
  Graphviz package:

  .. code-block:: shell

    dot -Tpng -o foo.png foo.dot

  .. versionadded:: 3.10
    The different dependency types ``PUBLIC``, ``INTERFACE`` and ``PRIVATE``
    are represented as solid, dashed and dotted edges.

  .. rubric:: Variables specific to the Graphviz support

  The resulting graphs can be huge.  The look and content of the generated graphs
  can be controlled using the file ``CMakeGraphVizOptions.cmake``.  This file is
  first searched in :variable:`CMAKE_BINARY_DIR`, and then in
  :variable:`CMAKE_SOURCE_DIR`.  If found, the variables set in it are used to
  adjust options for the generated Graphviz files.

  .. variable:: GRAPHVIZ_GRAPH_NAME

    The graph name.

    * Mandatory: NO
    * Default: value of :variable:`CMAKE_PROJECT_NAME`

  .. variable:: GRAPHVIZ_GRAPH_HEADER

    The header written at the top of the Graphviz files.

    * Mandatory: NO
    * Default: "node [ fontsize = "12" ];"

  .. variable:: GRAPHVIZ_NODE_PREFIX

    The prefix for each node in the Graphviz files.

    * Mandatory: NO
    * Default: "node"

  .. variable:: GRAPHVIZ_EXECUTABLES

    Set to FALSE to exclude executables from the generated graphs.

    * Mandatory: NO
    * Default: TRUE

  .. variable:: GRAPHVIZ_STATIC_LIBS

    Set to FALSE to exclude static libraries from the generated graphs.

    * Mandatory: NO
    * Default: TRUE

  .. variable:: GRAPHVIZ_SHARED_LIBS

    Set to FALSE to exclude shared libraries from the generated graphs.

    * Mandatory: NO
    * Default: TRUE

  .. variable:: GRAPHVIZ_MODULE_LIBS

    Set to FALSE to exclude module libraries from the generated graphs.

    * Mandatory: NO
    * Default: TRUE

  .. variable:: GRAPHVIZ_INTERFACE_LIBS

    Set to FALSE to exclude interface libraries from the generated graphs.

    * Mandatory: NO
    * Default: TRUE

  .. variable:: GRAPHVIZ_OBJECT_LIBS

    Set to FALSE to exclude object libraries from the generated graphs.

    * Mandatory: NO
    * Default: TRUE

  .. variable:: GRAPHVIZ_UNKNOWN_LIBS

    Set to FALSE to exclude unknown libraries from the generated graphs.

    * Mandatory: NO
    * Default: TRUE

  .. variable:: GRAPHVIZ_EXTERNAL_LIBS

    Set to FALSE to exclude external libraries from the generated graphs.

    * Mandatory: NO
    * Default: TRUE

  .. variable:: GRAPHVIZ_CUSTOM_TARGETS

    Set to TRUE to include custom targets in the generated graphs.

    * Mandatory: NO
    * Default: FALSE

  .. variable:: GRAPHVIZ_IGNORE_TARGETS

    A list of regular expressions for names of targets to exclude from the
    generated graphs.

    * Mandatory: NO
    * Default: empty

  .. variable:: GRAPHVIZ_GENERATE_PER_TARGET

    Set to FALSE to not generate per-target graphs ``foo.dot.<target>``.

    * Mandatory: NO
    * Default: TRUE

  .. variable:: GRAPHVIZ_GENERATE_DEPENDERS

    Set to FALSE to not generate depender graphs ``foo.dot.<target>.dependers``.

    * Mandatory: NO
    * Default: TRUE

.. option:: --system-information [file]

 Dump information about this system.

 Dump a wide range of information about the current system.  If run
 from the top of a binary tree for a CMake project it will dump
 additional information such as the cache, log files etc.

.. option:: --print-config-dir

 .. versionadded:: 3.31

 Print CMake config directory for user-wide FileAPI queries.

 See :envvar:`CMAKE_CONFIG_DIR` for more details.

.. option:: --log-level=<level>

 .. versionadded:: 3.16

 Set the log ``<level>``.

 The :command:`message` command will only output messages of the specified
 log level or higher.  The valid log levels are ``ERROR``, ``WARNING``,
 ``NOTICE``, ``STATUS`` (default), ``VERBOSE``, ``DEBUG``, or ``TRACE``.

 To make a log level persist between CMake runs, set
 :variable:`CMAKE_MESSAGE_LOG_LEVEL` as a cache variable instead.
 If both the command line option and the variable are given, the command line
 option takes precedence.

 For backward compatibility reasons, ``--loglevel`` is also accepted as a
 synonym for this option.

 .. versionadded:: 3.25
   See the :command:`cmake_language` command for a way to
   :ref:`query the current message logging level <query_message_log_level>`.

.. option:: --log-context

 Enable the :command:`message` command outputting context attached to each
 message.

 This option turns on showing context for the current CMake run only.
 To make showing the context persistent for all subsequent CMake runs, set
 :variable:`CMAKE_MESSAGE_CONTEXT_SHOW` as a cache variable instead.
 When this command line option is given, :variable:`CMAKE_MESSAGE_CONTEXT_SHOW`
 is ignored.

.. option:: --sarif-output=<path>

 .. versionadded:: 4.0

 Enable logging of diagnostic messages produced by CMake in the SARIF format.

 Write diagnostic messages to a SARIF file at the path specified. Projects can
 also set :variable:`CMAKE_EXPORT_SARIF` to ``ON`` to enable this feature for a
 build tree.

.. option:: --debug-trycompile

 Do not delete the files and directories created for
 :command:`try_compile` / :command:`try_run` calls.
 This is useful in debugging failed checks.

 Note that some uses of :command:`try_compile` may use the same build tree,
 which will limit the usefulness of this option if a project executes more
 than one :command:`try_compile`.  For example, such uses may change results
 as artifacts from a previous try-compile may cause a different test to either
 pass or fail incorrectly.  This option is best used only when debugging.

 (With respect to the preceding, the :command:`try_run` command
 is effectively a :command:`try_compile`.  Any combination of the two
 is subject to the potential issues described.)

 .. versionadded:: 3.25

   When this option is enabled, every try-compile check prints a log
   message reporting the directory in which the check is performed.

.. option:: --debug-output

 Put cmake in a debug mode.

 Print extra information during the cmake run like stack traces with
 :command:`message(SEND_ERROR)` calls.

.. option:: --debug-find

 .. versionadded:: 3.17

 Put cmake find commands in a debug mode.

 Print extra find call information during the cmake run to standard
 error. Output is designed for human consumption and not for parsing.
 See also the :variable:`CMAKE_FIND_DEBUG_MODE` variable for debugging
 a more local part of the project.

.. option:: --debug-find-pkg=<pkg>[,...]

 .. versionadded:: 3.23

 Put cmake find commands in a debug mode when running under calls
 to :command:`find_package(\<pkg\>) <find_package>`, where ``<pkg>``
 is an entry in the given comma-separated list of case-sensitive package
 names.

 Like :option:`--debug-find <cmake --debug-find>`, but limiting scope
 to the specified packages.

.. option:: --debug-find-var=<var>[,...]

 .. versionadded:: 3.23

 Put cmake find commands in a debug mode when called with ``<var>``
 as the result variable, where ``<var>`` is an entry in the given
 comma-separated list.

 Like :option:`--debug-find <cmake --debug-find>`, but limiting scope
 to the specified variable names.

.. option:: --trace

 Put cmake in trace mode.

 Print a trace of all calls made and from where.

.. option:: --trace-expand

 Put cmake in trace mode.

 Like :option:`--trace <cmake --trace>`, but with variables expanded.

.. option:: --trace-format=<format>

 .. versionadded:: 3.17

 Put cmake in trace mode and sets the trace output format.

 ``<format>`` can be one of the following values.

   ``human``
     Prints each trace line in a human-readable format. This is the
     default format.

   ``json-v1``
     Prints each line as a separate JSON document. Each document is
     separated by a newline (``\n``). It is guaranteed that no
     newline characters will be present inside a JSON document.

     .. code-block:: json
       :caption: JSON trace format

       {
         "file": "/full/path/to/the/CMake/file.txt",
         "line": 0,
         "cmd": "add_executable",
         "args": ["foo", "bar"],
         "time": 1579512535.9687231,
         "frame": 2,
         "global_frame": 4
       }

     The members are:

     ``file``
       The full path to the CMake source file where the function
       was called.

     ``line``
       The line in ``file`` where the function call begins.

     ``line_end``
       If the function call spans multiple lines, this field will
       be set to the line where the function call ends. If the function
       calls spans a single line, this field will be unset. This field
       was added in minor version 2 of the ``json-v1`` format.

     ``defer``
       Optional member that is present when the function call was deferred
       by :command:`cmake_language(DEFER)`.  If present, its value is a
       string containing the deferred call ``<id>``.

     ``cmd``
       The name of the function that was called.

     ``args``
       A string list of all function parameters.

     ``time``
       Timestamp (seconds since epoch) of the function call.

     ``frame``
       Stack frame depth of the function that was called, within the
       context of the  ``CMakeLists.txt`` being processed currently.

     ``global_frame``
       Stack frame depth of the function that was called, tracked globally
       across all ``CMakeLists.txt`` files involved in the trace. This field
       was added in minor version 2 of the ``json-v1`` format.

     Additionally, the first JSON document outputted contains the
     ``version`` key for the current major and minor version of the

     .. code-block:: json
       :caption: JSON version format

       {
         "version": {
           "major": 1,
           "minor": 2
         }
       }

     The members are:

     ``version``
       Indicates the version of the JSON format. The version has a
       major and minor components following semantic version conventions.

.. option:: --trace-source=<file>

 Put cmake in trace mode, but output only lines of a specified file.

 Multiple options are allowed.

.. option:: --trace-redirect=<file>

 Put cmake in trace mode and redirect trace output to a file instead of stderr.

.. option:: --warn-uninitialized

 Warn about uninitialized values.

 Print a warning when an uninitialized variable is used.

.. option:: --warn-unused-vars

 Does nothing.  In CMake versions 3.2 and below this enabled warnings about
 unused variables.  In CMake versions 3.3 through 3.18 the option was broken.
 In CMake 3.19 and above the option has been removed.

.. option:: --no-warn-unused-cli

 Don't warn about command line options.

 Don't find variables that are declared on the command line, but not
 used.

.. option:: --check-system-vars

 Find problems with variable usage in system files.

 Normally, unused and uninitialized variables are searched for only
 in :variable:`CMAKE_SOURCE_DIR` and :variable:`CMAKE_BINARY_DIR`.
 This flag tells CMake to warn about other files as well.

.. option:: --compile-no-warning-as-error

 .. versionadded:: 3.24

 Ignore target property :prop_tgt:`COMPILE_WARNING_AS_ERROR` and variable
 :variable:`CMAKE_COMPILE_WARNING_AS_ERROR`, preventing warnings from being
 treated as errors on compile.

.. option:: --link-no-warning-as-error

 .. versionadded:: 4.0

 Ignore target property :prop_tgt:`LINK_WARNING_AS_ERROR` and variable
 :variable:`CMAKE_LINK_WARNING_AS_ERROR`, preventing warnings from being
 treated as errors on link.

.. option:: --profiling-output=<path>

 .. versionadded:: 3.18

 Used in conjunction with
 :option:`--profiling-format <cmake --profiling-format>` to output to a
 given path.

.. option:: --profiling-format=<file>

 Enable the output of profiling data of CMake script in the given format.

 This can aid performance analysis of CMake scripts executed. Third party
 applications should be used to process the output into human readable format.

 Currently supported values are:
 ``google-trace`` Outputs in Google Trace Format, which can be parsed by the
 about:tracing tab of Google Chrome or using a plugin for a tool like Trace
 Compass.

.. option:: --preset <preset>, --preset=<preset>

 Reads a :manual:`preset <cmake-presets(7)>` from ``CMakePresets.json`` and
 ``CMakeUserPresets.json`` files, which must be located in the same directory
 as the top level ``CMakeLists.txt`` file. The preset may specify the
 generator, the build directory, a list of variables, and other arguments to
 pass to CMake. At least one of ``CMakePresets.json`` or
 ``CMakeUserPresets.json`` must be present.
 The :manual:`CMake GUI <cmake-gui(1)>` also recognizes and supports
 ``CMakePresets.json`` and ``CMakeUserPresets.json`` files. For full details
 on these files, see :manual:`cmake-presets(7)`.

 The presets are read before all other command line options, although the
 :option:`-S <cmake -S>` option can be used to specify the source directory
 containing the ``CMakePresets.json`` and ``CMakeUserPresets.json`` files.
 If :option:`-S <cmake -S>` is not given, the current directory is assumed to
 be the top level source directory and must contain the presets files. The
 options specified by the chosen preset (variables, generator, etc.) can all
 be overridden by manually specifying them on the command line. For example,
 if the preset sets a variable called ``MYVAR`` to ``1``, but the user sets
 it to ``2`` with a ``-D`` argument, the value ``2`` is preferred.

.. option:: --list-presets[=<type>]

 Lists the available presets of the specified ``<type>``.  Valid values for
 ``<type>`` are ``configure``, ``build``, ``test``, ``package``, or ``all``.
 If ``<type>`` is omitted, ``configure`` is assumed.  The current working
 directory must contain CMake preset files unless the :option:`-S <cmake -S>`
 option is used to specify a different top level source directory.

.. option:: --debugger

  Enables interactive debugging of the CMake language. CMake exposes a debugging
  interface on the pipe named by :option:`--debugger-pipe <cmake --debugger-pipe>`
  that conforms to the `Debug Adapter Protocol`_ specification with the following
  modifications.

  The ``initialize`` response includes an additional field named ``cmakeVersion``
  which specifies the version of CMake being debugged.

  .. code-block:: json
    :caption: Debugger initialize response

    {
      "cmakeVersion": {
        "major": 3,
        "minor": 27,
        "patch": 0,
        "full": "3.27.0"
      }
    }

  The members are:

  ``major``
    An integer specifying the major version number.

  ``minor``
    An integer specifying the minor version number.

  ``patch``
    An integer specifying the patch version number.

  ``full``
    A string specifying the full CMake version.

.. _`Debug Adapter Protocol`: https://microsoft.github.io/debug-adapter-protocol/

.. option:: --debugger-pipe <pipe name>, --debugger-pipe=<pipe name>

  Name of the pipe (on Windows) or domain socket (on Unix) to use for
  debugger communication.

.. option:: --debugger-dap-log <log path>, --debugger-dap-log=<log path>

  Logs all debugger communication to the specified file.

.. _`Build Tool Mode`:

Build a Project
===============

.. program:: cmake

CMake provides a command-line signature to build an already-generated
project binary tree:

.. code-block:: shell

  cmake --build <dir>             [<options>] [-- <build-tool-options>]
  cmake --build --preset <preset> [<options>] [-- <build-tool-options>]

This abstracts a native build tool's command-line interface with the
following options:

.. option:: --build <dir>

  Project binary directory to be built.  This is required (unless a preset
  is specified) and must be first.

.. program:: cmake--build

.. option:: --preset <preset>, --preset=<preset>

  Use a build preset to specify build options. The project binary directory
  is inferred from the ``configurePreset`` key. The current working directory
  must contain CMake preset files.
  See :manual:`preset <cmake-presets(7)>` for more details.

.. option:: --list-presets

  Lists the available build presets. The current working directory must
  contain CMake preset files.

.. option:: -j [<jobs>], --parallel [<jobs>]

  .. versionadded:: 3.12

  The maximum number of concurrent processes to use when building.
  If ``<jobs>`` is omitted the native build tool's default number is used.

  The :envvar:`CMAKE_BUILD_PARALLEL_LEVEL` environment variable, if set,
  specifies a default parallel level when this option is not given.

  Some native build tools always build in parallel.  The use of ``<jobs>``
  value of ``1`` can be used to limit to a single job.

.. option:: -t <tgt>..., --target <tgt>...

  Build ``<tgt>`` instead of the default target.  Multiple targets may be
  given, separated by spaces.

.. option:: --config <cfg>

  For multi-configuration tools, choose configuration ``<cfg>``.

.. option:: --clean-first

  Build target ``clean`` first, then build.
  (To clean only, use :option:`--target clean <cmake--build --target>`.)

.. option:: --resolve-package-references=<value>

  .. versionadded:: 3.23

  Resolve remote package references from external package managers (e.g. NuGet)
  before build. When ``<value>`` is set to ``on`` (default), packages will be
  restored before building a target.  When ``<value>`` is set to ``only``, the
  packages will be restored, but no build will be performed.  When
  ``<value>`` is set to ``off``, no packages will be restored.

  If the target does not define any package references, this option does nothing.

  This setting can be specified in a build preset (using
  ``resolvePackageReferences``). The preset setting will be ignored, if this
  command line option is specified.

  If no command line parameter or preset option are provided, an environment-
  specific cache variable will be evaluated to decide, if package restoration
  should be performed.

  When using :ref:`Visual Studio Generators`, package references are defined
  using the :prop_tgt:`VS_PACKAGE_REFERENCES` property. Package references
  are restored using NuGet. It can be disabled by setting the
  ``CMAKE_VS_NUGET_PACKAGE_RESTORE`` variable to ``OFF``.

.. option:: --use-stderr

  Ignored.  Behavior is default in CMake >= 3.0.

.. option:: -v, --verbose

  Enable verbose output - if supported - including the build commands to be
  executed.

  This option can be omitted if :envvar:`VERBOSE` environment variable or
  :variable:`CMAKE_VERBOSE_MAKEFILE` cached variable is set.


.. option:: --

  Pass remaining options to the native tool.

Run :option:`cmake --build` with no options for quick help.

Generator-Specific Build Tool Behavior
--------------------------------------

``cmake --build`` has special behavior with some generators:

:generator:`Xcode`

  .. versionadded:: 4.1

    If a third-party tool has written a ``.xcworkspace`` next to
    the CMake-generated ``.xcodeproj``, ``cmake --build`` drives
    the build through the workspace instead.

Install a Project
=================

.. program:: cmake

CMake provides a command-line signature to install an already-generated
project binary tree:

.. code-block:: shell

  cmake --install <dir> [<options>]

This may be used after building a project to run installation without
using the generated build system or the native build tool.
The options are:

.. option:: --install <dir>

  Project binary directory to install. This is required and must be first.

.. program:: cmake--install

.. option:: --config <cfg>

  For multi-configuration generators, choose configuration ``<cfg>``.

.. option:: --component <comp>

  Component-based install. Only install component ``<comp>``.

.. option:: --default-directory-permissions <permissions>

  Default directory install permissions. Permissions in format ``<u=rwx,g=rx,o=rx>``.

.. option:: --prefix <prefix>

  Override the installation prefix, :variable:`CMAKE_INSTALL_PREFIX`.

.. option:: --strip

  Strip before installing.

.. option:: -v, --verbose

  Enable verbose output.

  This option can be omitted if :envvar:`VERBOSE` environment variable is set.

.. option:: -j <jobs>, --parallel <jobs>

  .. versionadded:: 3.31

  Install in parallel using the given number of jobs. Only available if
  :prop_gbl:`INSTALL_PARALLEL` is enabled. The
  :envvar:`CMAKE_INSTALL_PARALLEL_LEVEL` environment variable specifies a
  default parallel level when this option is not provided.

Run :option:`cmake --install` with no options for quick help.

Open a Project
==============

.. program:: cmake

.. code-block:: shell

  cmake --open <dir>

Open the generated project in the associated application.  This is only
supported by some generators.


.. _`Script Processing Mode`:

Run a Script
============

.. program:: cmake

.. code-block:: shell

  cmake [-D <var>=<value>]... -P <cmake-script-file> [-- <unparsed-options>...]

.. program:: cmake-P

.. option:: -D <var>=<value>

 Define a variable for script mode.

.. program:: cmake

.. option:: -P <cmake-script-file>

 Process the given cmake file as a script written in the CMake
 language.  No configure or generate step is performed and the cache
 is not modified.  If variables are defined using ``-D``, this must be
 done before the ``-P`` argument.

Any options after ``--`` are not parsed by CMake, but they are still included
in the set of :variable:`CMAKE_ARGV<n> <CMAKE_ARGV0>` variables passed to the
script (including the ``--`` itself).


.. _`Run a Command-Line Tool`:

Run a Command-Line Tool
=======================

.. program:: cmake

CMake provides builtin command-line tools through the signature

.. code-block:: shell

  cmake -E <command> [<options>]

.. option:: -E [help]

  Run ``cmake -E`` or ``cmake -E help`` for a summary of commands.

.. program:: cmake-E

Available commands are:

.. option:: capabilities

  .. versionadded:: 3.7

  Report cmake capabilities in JSON format. The output is a JSON object
  with the following keys:

  ``version``
    A JSON object with version information. Keys are:

    ``string``
      The full version string as displayed by cmake :option:`--version <cmake --version>`.
    ``major``
      The major version number in integer form.
    ``minor``
      The minor version number in integer form.
    ``patch``
      The patch level in integer form.
    ``suffix``
      The cmake version suffix string.
    ``isDirty``
      A bool that is set if the cmake build is from a dirty tree.

  ``generators``
    A list available generators. Each generator is a JSON object with the
    following keys:

    ``name``
      A string containing the name of the generator.
    ``toolsetSupport``
      ``true`` if the generator supports toolsets and ``false`` otherwise.
    ``platformSupport``
      ``true`` if the generator supports platforms and ``false`` otherwise.
    ``supportedPlatforms``
      .. versionadded:: 3.21

      Optional member that may be present when the generator supports
      platform specification via :variable:`CMAKE_GENERATOR_PLATFORM`
      (:option:`-A ... <cmake -A>`).  The value is a list of platforms known to
      be supported.
    ``extraGenerators``
      A list of strings with all the :ref:`Extra Generators` compatible with
      the generator.

  ``fileApi``
    Optional member that is present when the :manual:`cmake-file-api(7)`
    is available.  The value is a JSON object with one member:

    ``requests``
      A JSON array containing zero or more supported file-api requests.
      Each request is a JSON object with members:

      ``kind``
        Specifies one of the supported :ref:`file-api object kinds`.

      ``version``
        A JSON array whose elements are each a JSON object containing
        ``major`` and ``minor`` members specifying non-negative integer
        version components.

  ``serverMode``
    ``true`` if cmake supports server-mode and ``false`` otherwise.
    Always false since CMake 3.20.

  ``tls``
    .. versionadded:: 3.25

    ``true`` if TLS support is enabled and ``false`` otherwise.

  ``debugger``
    .. versionadded:: 3.27

    ``true`` if the :option:`--debugger <cmake --debugger>` mode
    is supported and ``false`` otherwise.

.. option:: cat [--] <files>...

  .. versionadded:: 3.18

  Concatenate files and print on the standard output.

  .. program:: cmake-E_cat

  .. option:: --

    .. versionadded:: 3.24

    Added support for the double dash argument ``--``. This basic implementation
    of ``cat`` does not support any options, so using a option starting with
    ``-`` will result in an error. Use ``--`` to indicate the end of options, in
    case a file starts with ``-``.

  .. versionadded:: 3.29

    ``cat`` can now print the standard input by passing the ``-`` argument.

.. program:: cmake-E

.. option:: chdir <dir> <cmd> [<arg>...]

  Change the current working directory and run a command.

.. option:: compare_files [--ignore-eol] <file1> <file2>

  Check if ``<file1>`` is same as ``<file2>``. If files are the same,
  then returns ``0``, if not it returns ``1``.  In case of invalid
  arguments, it returns 2.

  .. program:: cmake-E_compare_files

  .. option:: --ignore-eol

    .. versionadded:: 3.14

    The option implies line-wise comparison and ignores LF/CRLF differences.

.. program:: cmake-E

.. option:: copy <file>... <destination>, copy -t <destination> <file>...

  Copy files to ``<destination>`` (either file or directory).
  If multiple files are specified, or if ``-t`` is specified, the
  ``<destination>`` must be directory and it must exist. If ``-t`` is not
  specified, the last argument is assumed to be the ``<destination>``.
  Wildcards are not supported. ``copy`` does follow symlinks. That means it
  does not copy symlinks, but the files or directories it point to.

  .. versionadded:: 3.5
    Support for multiple input files.

  .. versionadded:: 3.26
    Support for ``-t`` argument.

.. option:: copy_directory <dir>... <destination>

  Copy content of ``<dir>...`` directories to ``<destination>`` directory.
  If ``<destination>`` directory does not exist it will be created.
  ``copy_directory`` does follow symlinks.

  .. versionadded:: 3.5
    Support for multiple input directories.

  .. versionadded:: 3.15
    The command now fails when the source directory does not exist.
    Previously it succeeded by creating an empty destination directory.

.. option:: copy_directory_if_different <dir>... <destination>

  .. versionadded:: 3.26

  Copy changed content of ``<dir>...`` directories to ``<destination>`` directory.
  If ``<destination>`` directory does not exist it will be created.

  ``copy_directory_if_different`` does follow symlinks.
  The command fails when the source directory does not exist.

.. option:: copy_directory_if_newer <dir>... <destination>

  .. versionadded:: 4.2

  Copy content of ``<dir>...`` directories to ``<destination>`` directory
  if source files are newer than destination files (based on file timestamps).
  If ``<destination>`` directory does not exist it will be created.

  ``copy_directory_if_newer`` does follow symlinks.
  The command fails when the source directory does not exist.
  This is faster than ``copy_directory_if_different`` as it only compares
  file timestamps instead of file contents.

.. option:: copy_if_different <file>... <destination>

  Copy files to ``<destination>`` (either file or directory) if
  they have changed.
  If multiple files are specified, the ``<destination>`` must be
  directory and it must exist.
  ``copy_if_different`` does follow symlinks.

  .. versionadded:: 3.5
    Support for multiple input files.

.. option:: copy_if_newer <file>... <destination>

  .. versionadded:: 4.2

  Copy files to ``<destination>`` (either file or directory) if
  source files are newer than destination files (based on file timestamps).
  If multiple files are specified, the ``<destination>`` must be
  directory and it must exist.
  ``copy_if_newer`` does follow symlinks.
  This is faster than ``copy_if_different`` as it only compares
  file timestamps instead of file contents.

.. option:: create_symlink <old> <new>

  Create a symbolic link ``<new>`` naming ``<old>``.

  .. versionadded:: 3.13
    Support for creating symlinks on Windows.

  .. note::
    Path to where ``<new>`` symbolic link will be created has to exist beforehand.

.. option:: create_hardlink <old> <new>

  .. versionadded:: 3.19

  Create a hard link ``<new>`` naming ``<old>``.

  .. note::
    Path to where ``<new>`` hard link will be created has to exist beforehand.
    ``<old>`` has to exist beforehand.

.. option:: echo [<string>...]

  Displays arguments as text.

.. option:: echo_append [<string>...]

  Displays arguments as text but no new line.

.. option:: env [<options>] [--] <command> [<arg>...]

  .. versionadded:: 3.1

  Run command in a modified environment. Options are:

  .. program:: cmake-E_env

  .. option:: NAME=VALUE

    Replaces the current value of ``NAME`` with ``VALUE``.

  .. option:: --unset=NAME

    Unsets the current value of ``NAME``.

  .. option:: --modify ENVIRONMENT_MODIFICATION

    .. versionadded:: 3.25

    Apply a single :prop_test:`ENVIRONMENT_MODIFICATION` operation to the
    modified environment.

    The ``NAME=VALUE`` and ``--unset=NAME`` options are equivalent to
    ``--modify NAME=set:VALUE`` and ``--modify NAME=unset:``, respectively.
    Note that ``--modify NAME=reset:`` resets ``NAME`` to the value it had
    when :program:`cmake` launched (or unsets it), not to the most recent
    ``NAME=VALUE`` option.

  .. option:: --

    .. versionadded:: 3.24

    Added support for the double dash argument ``--``. Use ``--`` to stop
    interpreting options/environment variables and treat the next argument as
    the command, even if it start with ``-`` or contains a ``=``.

.. program:: cmake-E

.. option:: environment

  Display the current environment variables.

.. option:: false

  .. versionadded:: 3.16

  Do nothing, with an exit code of 1.

.. option:: make_directory <dir>...

  Create ``<dir>`` directories.  If necessary, create parent
  directories too.  If a directory already exists it will be
  silently ignored.

  .. versionadded:: 3.5
    Support for multiple input directories.

.. option:: md5sum <file>...

  Create MD5 checksum of files in ``md5sum`` compatible format::

     351abe79cd3800b38cdfb25d45015a15  file1.txt
     052f86c15bbde68af55c7f7b340ab639  file2.txt

.. option:: sha1sum <file>...

  .. versionadded:: 3.10

  Create SHA1 checksum of files in ``sha1sum`` compatible format::

     4bb7932a29e6f73c97bb9272f2bdc393122f86e0  file1.txt
     1df4c8f318665f9a5f2ed38f55adadb7ef9f559c  file2.txt

.. option:: sha224sum <file>...

  .. versionadded:: 3.10

  Create SHA224 checksum of files in ``sha224sum`` compatible format::

     b9b9346bc8437bbda630b0b7ddfc5ea9ca157546dbbf4c613192f930  file1.txt
     6dfbe55f4d2edc5fe5c9197bca51ceaaf824e48eba0cc453088aee24  file2.txt

.. option:: sha256sum <file>...

  .. versionadded:: 3.10

  Create SHA256 checksum of files in ``sha256sum`` compatible format::

     76713b23615d31680afeb0e9efe94d47d3d4229191198bb46d7485f9cb191acc  file1.txt
     15b682ead6c12dedb1baf91231e1e89cfc7974b3787c1e2e01b986bffadae0ea  file2.txt

.. option:: sha384sum <file>...

  .. versionadded:: 3.10

  Create SHA384 checksum of files in ``sha384sum`` compatible format::

     acc049fedc091a22f5f2ce39a43b9057fd93c910e9afd76a6411a28a8f2b8a12c73d7129e292f94fc0329c309df49434  file1.txt
     668ddeb108710d271ee21c0f3acbd6a7517e2b78f9181c6a2ff3b8943af92b0195dcb7cce48aa3e17893173c0a39e23d  file2.txt

.. option:: sha512sum <file>...

  .. versionadded:: 3.10

  Create SHA512 checksum of files in ``sha512sum`` compatible format::

     2a78d7a6c5328cfb1467c63beac8ff21794213901eaadafd48e7800289afbc08e5fb3e86aa31116c945ee3d7bf2a6194489ec6101051083d1108defc8e1dba89  file1.txt
     7a0b54896fe5e70cca6dd643ad6f672614b189bf26f8153061c4d219474b05dad08c4e729af9f4b009f1a1a280cb625454bf587c690f4617c27e3aebdf3b7a2d  file2.txt

.. option:: remove [-f] <file>...

  .. deprecated:: 3.17

  Remove the file(s). The planned behavior was that if any of the
  listed files already do not exist, the command returns a non-zero exit code,
  but no message is logged. The ``-f`` option changes the behavior to return a
  zero exit code (i.e. success) in such situations instead.
  ``remove`` does not follow symlinks. That means it remove only symlinks
  and not files it point to.

  The implementation was buggy and always returned 0. It cannot be fixed without
  breaking backwards compatibility. Use ``rm`` instead.

.. option:: remove_directory <dir>...

  .. deprecated:: 3.17

  Remove ``<dir>`` directories and their contents. If a directory does
  not exist it will be silently ignored.
  Use ``rm`` instead.

  .. versionadded:: 3.15
    Support for multiple directories.

  .. versionadded:: 3.16
    If ``<dir>`` is a symlink to a directory, just the symlink will be removed.

.. option:: rename <oldname> <newname>

  Rename a file or directory (on one volume). If file with the ``<newname>`` name
  already exists, then it will be silently replaced.

.. option:: rm [-rRf] [--] <file|dir>...

  .. versionadded:: 3.17

  Remove the files ``<file>`` or directories ``<dir>``.
  Use ``-r`` or ``-R`` to remove directories and their contents recursively.
  If any of the listed files/directories do not exist, the command returns a
  non-zero exit code, but no message is logged. The ``-f`` option changes
  the behavior to return a zero exit code (i.e. success) in such
  situations instead. Use ``--`` to stop interpreting options and treat all
  remaining arguments as paths, even if they start with ``-``.

.. option:: sleep <number>

  .. versionadded:: 3.0

  Sleep for ``<number>`` seconds. ``<number>`` may be a floating point number.
  A practical minimum is about 0.1 seconds due to overhead in starting/stopping
  CMake executable. This can be useful in a CMake script to insert a delay:

  .. code-block:: cmake

    # Sleep for about 0.5 seconds
    execute_process(COMMAND ${CMAKE_COMMAND} -E sleep 0.5)

.. option:: tar [cxt][vf][zjJ] file.tar [<options>] [--] [<pathname>...]

  Create or extract a tar or zip archive.  Options are:

  .. program:: cmake-E_tar

  .. option:: c

    Create a new archive containing the specified files.
    If used, the ``<pathname>...`` argument is mandatory.

  .. option:: x

    Extract to disk from the archive.

    .. versionadded:: 3.15
      The ``<pathname>...`` argument could be used to extract only selected files
      or directories.
      When extracting selected files or directories, you must provide their exact
      names including the path, as printed by list (``-t``).

  .. option:: t

    List archive contents.

    .. versionadded:: 3.15
      The ``<pathname>...`` argument could be used to list only selected files
      or directories.

  .. option:: v

    Produce verbose output.

  .. option:: z

    Compress the resulting archive with gzip.

  .. option:: j

    Compress the resulting archive with bzip2.

  .. option:: J

    .. versionadded:: 3.1

    Compress the resulting archive with XZ.

  .. option:: --zstd

    .. versionadded:: 3.15

    Compress the resulting archive with Zstandard.

  .. option:: --files-from=<file>

    .. versionadded:: 3.1

    Read file names from the given file, one per line.
    Blank lines are ignored.  Lines may not start in ``-``
    except for ``--add-file=<name>`` to add files whose
    names start in ``-``.

  .. option:: --format=<format>

    .. versionadded:: 3.3

    Specify the format of the archive to be created.
    Supported formats are: ``7zip``, ``gnutar``, ``pax``,
    ``paxr`` (restricted pax, default), and ``zip``.

  .. option:: --mtime=<date>

    .. versionadded:: 3.1

    Specify modification time recorded in tarball entries.

  .. option:: --touch

    .. versionadded:: 3.24

    Use current local timestamp instead of extracting file timestamps
    from the archive.

  .. option:: --

    .. versionadded:: 3.1

    Stop interpreting options and treat all remaining arguments
    as file names, even if they start with ``-``.

  .. versionadded:: 3.1
    LZMA (7zip) support.

  .. versionadded:: 3.15
    The command now continues adding files to an archive even if some of the
    files are not readable.  This behavior is more consistent with the classic
    ``tar`` tool. The command now also parses all flags, and if an invalid flag
    was provided, a warning is issued.

.. program:: cmake-E

.. option:: time <command> [<args>...]

  Run ``<command>`` and display elapsed time (including overhead of CMake frontend).

  .. versionadded:: 3.5
    The command now properly passes arguments with spaces or special characters
    through to the child process. This may break scripts that worked around the
    bug with their own extra quoting or escaping.

.. option:: touch <file>...

  Creates ``<file>`` if file do not exist.
  If ``<file>`` exists, it is changing ``<file>`` access and modification times.

.. option:: touch_nocreate <file>...

  Touch a file if it exists but do not create it.  If a file does
  not exist it will be silently ignored.

.. option:: true

  .. versionadded:: 3.16

  Do nothing, with an exit code of 0.

Windows-specific Command-Line Tools
-----------------------------------

The following ``cmake -E`` commands are available only on Windows:

.. option:: delete_regv <key>

  Delete Windows registry value.

.. option:: env_vs8_wince <sdkname>

  .. versionadded:: 3.2

  Displays a batch file which sets the environment for the provided
  Windows CE SDK installed in VS2005.

.. option:: env_vs9_wince <sdkname>

  .. versionadded:: 3.2

  Displays a batch file which sets the environment for the provided
  Windows CE SDK installed in VS2008.

.. option:: write_regv <key> <value>

  Write Windows registry value.

.. _`Find-Package Tool Mode`:

Run the Find-Package Tool
=========================

.. program:: cmake--find-package

CMake provides a pkg-config like helper for Makefile-based projects:

.. code-block:: shell

  cmake --find-package [<options>]

.. note::
  This mode is not well-supported due to some technical limitations.
  It is kept for compatibility but should not be used in new projects.

.. option:: --find-package

  It searches a package using the :command:`find_package` command and prints the
  resulting flags to stdout.  This can be used instead of pkg-config to find
  installed libraries in plain Makefile-based projects or in Autoconf-based
  projects, using auxiliary macros installed in ``share/aclocal/cmake.m4`` on
  the system.

  When using this option, the following variables are expected:

  ``NAME``
    Name of the package as called in ``find_package(<PackageName>)``.

  ``COMPILER_ID``
    :variable:`Compiler ID <CMAKE_<LANG>_COMPILER_ID>` used for searching the
    package, i.e. GNU/Intel/Clang/MSVC, etc.

  ``LANGUAGE``
    Language used for searching the package, i.e. C/CXX/Fortran/ASM, etc.

  ``MODE``
    The package search mode.  Value can be one of:

    ``EXIST``
      Only checks for existence of the given package.

    ``COMPILE``
      Prints the flags needed for compiling an object file which uses the given
      package.

    ``LINK``
      Prints the flags needed for linking when using the given package.

  ``SILENT``
    (Optional) If TRUE, find result message is not printed.

  For example:

  .. code-block:: shell

    cmake --find-package -DNAME=CURL -DCOMPILER_ID=GNU -DLANGUAGE=C -DMODE=LINK

.. _`Workflow Mode`:

Run a Workflow Preset
=====================

.. versionadded:: 3.25

.. program:: cmake

:manual:`CMake Presets <cmake-presets(7)>` provides a way to execute multiple
build steps in order:

.. code-block:: shell

  cmake --workflow <options>

The options are:

.. option:: --workflow

  Select a :ref:`Workflow Preset` using one of the following options.

.. program:: cmake--workflow

.. option:: --preset <preset>, --preset=<preset>

  Use a workflow preset to specify a workflow. The project binary directory
  is inferred from the initial configure preset. The current working directory
  must contain CMake preset files.
  See :manual:`preset <cmake-presets(7)>` for more details.

  .. versionchanged:: 3.31
    When following immediately after the ``--workflow`` option,
    the ``--preset`` argument can be omitted and just the ``<preset>``
    name can be given.  This means the following syntax is valid:

    .. code-block:: console

      $ cmake --workflow my-preset

.. option:: --list-presets

  Lists the available workflow presets. The current working directory must
  contain CMake preset files.

.. option:: --fresh

  Perform a fresh configuration of the build tree, which has the same effect
  as :option:`cmake --fresh`.

View Help
=========

.. program:: cmake

To print selected pages from the CMake documentation, use

.. code-block:: shell

  cmake --help[-<topic>]

with one of the following options:

.. include:: include/OPTIONS_HELP.rst

To view the presets available for a project, use

.. code-block:: shell

  cmake <source-dir> --list-presets


.. _`CMake Exit Code`:

Return Value (Exit Code)
========================

Upon regular termination, the :program:`cmake` executable returns the exit code ``0``.

If termination is caused by the command :command:`message(FATAL_ERROR)`,
or another error condition, then a non-zero exit code is returned.


See Also
========

.. include:: include/LINKS.rst
