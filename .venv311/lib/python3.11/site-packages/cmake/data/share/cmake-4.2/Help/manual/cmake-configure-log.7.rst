.. cmake-manual-description: CMake Configure Log

cmake-configure-log(7)
**********************

.. versionadded:: 3.26

.. only:: html

   .. contents::

Introduction
============

CMake writes a running log, known as the *configure log*,
of certain events that occur during the Configure step.
The configure log does *not* contain a log of all output, errors,
or messages printed while configuring a project.  It is a log of
detailed information about specific events, such as toolchain inspection
by :command:`try_compile`, meant for use in debugging the configuration
of a build tree.

For human use, this version of CMake writes the configure log to the file:

.. code-block:: cmake

  ${CMAKE_BINARY_DIR}/CMakeFiles/CMakeConfigureLog.yaml

However, the *location and name of the log file may change* in future
versions of CMake.  Tools that read the configure log should get its
location using a :ref:`configureLog <file-api configureLog>` query to
the :manual:`cmake-file-api(7)`.
See the `Log Versioning`_ section below for details.

Log Structure
=============

The configure log is designed to be both machine- and human-readable.

The log file is a YAML document stream containing zero or more YAML
documents separated by document markers.  Each document begins
with a ``---`` document marker line, contains a single YAML mapping
that logs events from one CMake "configure" step, and, if the configure
step finished normally, ends with a ``...`` document marker line:

.. code-block:: yaml

  ---
  events:
    -
      kind: "try_compile-v1"
      # (other fields omitted)
    -
      kind: "try_compile-v1"
      # (other fields omitted)
  ...

A new document is appended to the log every time CMake configures
the build tree and logs new events.

The keys of each document root mapping are:

``events``
  A YAML block sequence of nodes corresponding to events logged during
  one CMake "configure" step.  Each event is a YAML node containing one
  of the `Event Kinds`_ documented below.

Log Versioning
--------------

Each of the `Event Kinds`_ is versioned independently.  The set of
keys an event's log entry provides is specific to its major version.
When an event is logged, the latest version of its event kind that is
known to the running version of CMake is always written to the log.

Tools reading the configure log must ignore event kinds and versions
they do not understand:

* A future version of CMake may introduce a new event kind or version.

* If an existing build tree is re-configured with a different version of
  CMake, the log may contain different versions of the same event kind.

* If :manual:`cmake-file-api(7)` queries request one or more
  :ref:`configureLog <file-api configureLog>` object versions,
  the log may contain multiple entries for the same event, each
  with a different version of its event kind.

IDEs should write a :manual:`cmake-file-api(7)` query requesting a
specific :ref:`configureLog <file-api configureLog>` object version,
before running CMake, and then read the configure log only as described
by the file-api reply.

Text Block Encoding
-------------------

In order to make the log human-readable, text blocks are always
represented using YAML literal block scalars (``|``).
Since literal block scalars do not support escaping, backslashes
and non-printable characters are encoded at the application layer:

* ``\\`` encodes a backslash.
* ``\xXX`` encodes a byte using two hexadecimal digits, ``XX``.

.. _`configure-log event kinds`:

Event Kinds
===========

Every event kind is represented by a YAML mapping of the form:

.. code-block:: yaml

  kind: "<kind>-v<major>"
  backtrace:
    - "<file>:<line> (<function>)"
  checks:
    - "Checking for something"
  #...event-specific keys...

The keys common to all events are:

``kind``
  A string identifying the event kind and major version.

``backtrace``
  A YAML block sequence reporting the call stack of CMake source
  locations at which the event occurred, from most-recent to
  least-recent.  Each node is a string specifying one location
  formatted as ``<file>:<line> (<function>)``.

``checks``
  An optional key that is present when the event occurred with
  at least one pending :command:`message(CHECK_START)`.  Its value
  is a YAML block sequence reporting the stack of pending checks,
  from most-recent to least-recent.  Each node is a string containing
  a pending check message.

Additional mapping keys are specific to each (versioned) event kind,
described below.

.. _`message configure-log event`:

Event Kind ``message``
----------------------

The :command:`message(CONFIGURE_LOG)` command logs ``message`` events.

There is only one ``message`` event major version, version 1.

.. _`message-v1 event`:

``message-v1`` Event
^^^^^^^^^^^^^^^^^^^^

A ``message-v1`` event is a YAML mapping:

.. code-block:: yaml

  kind: "message-v1"
  backtrace:
    - "CMakeLists.txt:123 (message)"
  checks:
    - "Checking for something"
  message: |
    # ...

The keys specific to ``message-v1`` mappings are:

``message``
  A YAML literal block scalar containing the message text,
  represented using our `Text Block Encoding`_.

.. _`try_compile configure-log event`:

Event Kind ``try_compile``
--------------------------

The :command:`try_compile` command logs ``try_compile`` events.

There is only one ``try_compile`` event major version, version 1.

.. _`try_compile-v1 event`:

``try_compile-v1`` Event
^^^^^^^^^^^^^^^^^^^^^^^^

A ``try_compile-v1`` event is a YAML mapping:

.. code-block:: yaml

  kind: "try_compile-v1"
  backtrace:
    - "CMakeLists.txt:123 (try_compile)"
  checks:
    - "Checking for something"
  description: "Explicit LOG_DESCRIPTION"
  directories:
    source: "/path/to/.../TryCompile-01234"
    binary: "/path/to/.../TryCompile-01234"
  cmakeVariables:
    SOME_VARIABLE: "Some Value"
  buildResult:
    variable: "COMPILE_RESULT"
    cached: true
    stdout: |
      # ...
    exitCode: 0

The keys specific to ``try_compile-v1`` mappings are:

``description``
  An optional key that is present when the ``LOG_DESCRIPTION <text>`` option
  was used.  Its value is a string containing the description ``<text>``.

``directories``
  A mapping describing the directories associated with the
  compilation attempt.  It has the following keys:

  ``source``
    String specifying the source directory of the
    :command:`try_compile` project.

  ``binary``
    String specifying the binary directory of the
    :command:`try_compile` project.
    For non-project invocations, this is often the same as
    the source directory.

``cmakeVariables``
  An optional key that is present when CMake propagates variables
  into the test project, either automatically or due to the
  :variable:`CMAKE_TRY_COMPILE_PLATFORM_VARIABLES` variable.
  Its value is a mapping from variable names to their values.

``buildResult``
  A mapping describing the result of compiling the test code.
  It has the following keys:

  ``variable``
    A string specifying the name of the CMake variable
    storing the result of trying to build the test project.

  ``cached``
    A boolean indicating whether the above result ``variable``
    is stored in the CMake cache.

  ``stdout``
    A YAML literal block scalar containing the output from building
    the test project, represented using our `Text Block Encoding`_.
    This contains build output from both stdout and stderr.

  ``exitCode``
    An integer specifying the build tool exit code from trying
    to build the test project.

.. _`try_run configure-log event`:

Event Kind ``try_run``
----------------------

The :command:`try_run` command logs ``try_run`` events.

There is only one ``try_run`` event major version, version 1.

.. _`try_run-v1 event`:

``try_run-v1`` Event
^^^^^^^^^^^^^^^^^^^^

A ``try_run-v1`` event is a YAML mapping:

.. code-block:: yaml

  kind: "try_run-v1"
  backtrace:
    - "CMakeLists.txt:456 (try_run)"
  checks:
    - "Checking for something"
  description: "Explicit LOG_DESCRIPTION"
  directories:
    source: "/path/to/.../TryCompile-56789"
    binary: "/path/to/.../TryCompile-56789"
  buildResult:
    variable: "COMPILE_RESULT"
    cached: true
    stdout: |
      # ...
    exitCode: 0
  runResult:
    variable: "RUN_RESULT"
    cached: true
    stdout: |
      # ...
    stderr: |
      # ...
    exitCode: 0

The keys specific to ``try_run-v1`` mappings include those
documented by the `try_compile-v1 event`_, plus:

``runResult``
  A mapping describing the result of running the test code.
  It has the following keys:

  ``variable``
    A string specifying the name of the CMake variable
    storing the result of trying to run the test executable.

  ``cached``
    A boolean indicating whether the above result ``variable``
    is stored in the CMake cache.

  ``stdout``
    An optional key that is present when the test project built successfully.
    Its value is a YAML literal block scalar containing output from running
    the test executable, represented using our `Text Block Encoding`_.

    If ``RUN_OUTPUT_VARIABLE`` was used, stdout and stderr are captured
    together, so this will contain both.  Otherwise, this will contain
    only the stdout output.

  ``stderr``
    An optional key that is present when the test project built successfully
    and the ``RUN_OUTPUT_VARIABLE`` option was not used.
    Its value is a YAML literal block scalar containing output from running
    the test executable, represented using our `Text Block Encoding`_.

    If ``RUN_OUTPUT_VARIABLE`` was used, stdout and stderr are captured
    together in the ``stdout`` key, and this key will not be present.
    Otherwise, this will contain the stderr output.

  ``exitCode``
    An optional key that is present when the test project built successfully.
    Its value is an integer specifying the exit code, or a string containing
    an error message, from trying to run the test executable.

.. _`find configure-log event`:

Event Kind ``find``
-------------------

The :command:`find_file`, :command:`find_path`, :command:`find_library`, and
:command:`find_program` commands log ``find`` events.

There is only one ``find`` event major version, version 1.

.. _`find-v1 event`:

``find-v1`` Event
^^^^^^^^^^^^^^^^^

.. versionadded:: 4.1

A ``find-v1`` event is a YAML mapping:

.. code-block:: yaml

  kind: "find-v1"
  backtrace:
    - "CMakeLists.txt:456 (find_program)"
  mode: "program"
  variable: "PROGRAM_PATH"
  description: "Docstring for variable"
  settings:
    SearchFramework: "NEVER"
    SearchAppBundle: "NEVER"
    CMAKE_FIND_USE_CMAKE_PATH: true
    CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH: true
    CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH: true
    CMAKE_FIND_USE_CMAKE_SYSTEM_PATH: true
    CMAKE_FIND_USE_INSTALL_PREFIX: true
  names:
    - "name1"
    - "name2"
  candidate_directories:
    - "/path/to/search"
    - "/other/path/to/search"
    - "/path/to/found"
    - "/further/path/to/search"
  searched_directories:
    - "/path/to/search"
    - "/other/path/to/search"
  found: "/path/to/found/program"

The keys specific to ``find-v1`` mappings are:

``mode``
  A string describing the command using the search performed. One of ``file``,
  ``path``, ``program``, or ``library``.

``variable``
  The variable to which the search stored its result.

``description``
  The documentation string of the variable.

``settings``
  Search settings active for the search.

  ``SearchFramework``
    A string describing how framework search is performed. One of ``FIRST``,
    ``LAST``, ``ONLY``, or ``NEVER``. See :variable:`CMAKE_FIND_FRAMEWORK`.

  ``SearchAppBundle``
    A string describing how application bundle search is performed. One of
    ``FIRST``, ``LAST``, ``ONLY``, or ``NEVER``. See
    :variable:`CMAKE_FIND_APPBUNDLE`.

  ``CMAKE_FIND_USE_CMAKE_PATH``
    A boolean indicating whether or not CMake-specific cache variables are
    used when searching. See :variable:`CMAKE_FIND_USE_CMAKE_PATH`.

  ``CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH``
    A boolean indicating whether or not CMake-specific environment variables
    are used when searching. See
    :variable:`CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH`.

  ``CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH``
    A boolean indicating whether or not platform-specific environment
    variables are used when searching. See
    :variable:`CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH`.

  ``CMAKE_FIND_USE_CMAKE_SYSTEM_PATH``
    A boolean indicating whether or not platform-specific CMake variables are
    used when searching. See :variable:`CMAKE_FIND_USE_CMAKE_SYSTEM_PATH`.

  ``CMAKE_FIND_USE_INSTALL_PREFIX``
    A boolean indicating whether or not the install prefix is used when
    searching. See :variable:`CMAKE_FIND_USE_INSTALL_PREFIX`.

``names``
  The names to look for the queries.

``candidate_directories``
  Candidate directories, in order, to look in during the search.

``searched_directories``
  Directories, in order, looked at during the search process.

``found``
  Either a string representing the found value or ``false`` if it was not
  found.

``search_context``
  A mapping of variable names to search paths specified by them (either a
  string or an array of strings depending on the variable). Environment
  variables are wrapped with ``ENV{`` and ``}``, otherwise CMake variables are
  used. Only variables with any paths specified are used.

  ``package_stack``
    An array of objects with paths which come from the stack of paths made
    available by :command:`find_package` calls.

    ``package_paths``
      The paths made available by :command:`find_package` commands in the call
      stack.

.. _`find_package configure-log event`:

Event Kind ``find_package``
---------------------------

.. versionadded:: 4.1

The :command:`find_package` command logs ``find_package`` events.

There is only one ``find_package`` event major version, version 1.

.. _`find_package-v1 event`:

``find_package-v1`` Event
^^^^^^^^^^^^^^^^^^^^^^^^^

A ``find_package-v1`` event is a YAML mapping:

.. code-block:: yaml

  kind: "find_package-v1"
  backtrace:
    - "CMakeLists.txt:456 (find_program)"
  name: "PackageName"
  components:
    -
      name: "Component"
      required: true
      found: true
  configs:
    -
      filename: PackageNameConfig.cmake
      kind: "cmake"
    -
      filename: packagename-config.cmake
      kind: "cmake"
  version_request:
    version: "1.0"
    version_complete: "1.0...1.5"
    min: "INCLUDE"
    max: "INCLUDE"
    exact: false
  settings:
    required: "optional"
    quiet: false
    global: false
    policy_scope: true
    bypass_provider: false
    hints:
      - "/hint/path"
    names:
      - "name1"
      - "name2"
    search_paths:
      - "/search/path"
    path_suffixes:
      - ""
      - "suffix"
    registry_view: "HOST"
    paths:
      CMAKE_FIND_USE_CMAKE_PATH: true
      CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH: true
      CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH: true
      CMAKE_FIND_USE_CMAKE_SYSTEM_PATH: true
      CMAKE_FIND_USE_INSTALL_PREFIX: true
      CMAKE_FIND_USE_PACKAGE_ROOT_PATH: true
      CMAKE_FIND_USE_CMAKE_PACKAGE_REGISTRY: true
      CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY: true
      CMAKE_FIND_ROOT_PATH_MODE: "BOTH"
    candidates:
      -
        path: "/path/to/config/PackageName/PackageNameConfig.cmake"
        mode: "config"
        reason: "insufficient_version"
      -
        path: "/path/to/config/PackageName/packagename-config.cmake"
        mode: "config"
        reason: "no_exist"
    found:
      path: "/path/to/config/PackageName-2.5/PackageNameConfig.cmake"
      mode: "config"
      version: "2.5"

The keys specific to ``find_package-v1`` mappings are:

``name``
  The name of the requested package.

``components``
  If present, an array of objects containing the fields:

  ``name``
    The name of the component.

  ``required``
    A boolean indicating whether the component is required or optional.

  ``found``
    A boolean indicating whether the component was found or not.

``configs``
  If present, an array of objects indicating the configuration files to search
  for.

  ``filename``
    The filename of the configuration file.

  ``kind``
    The kind of file. Either ``cmake`` or ``cps``.

``version_request``
  An object indicating the version constraints on the search.

  ``version``
    The minimum version required.

  ``version_complete``
    The user-provided version range.

  ``min``
    Whether to ``INCLUDE`` or ``EXCLUDE`` the lower bound on the version
    range.

  ``max``
    Whether to ``INCLUDE`` or ``EXCLUDE`` the upper bound on the version
    range.

  ``exact``
    A boolean indicating whether an ``EXACT`` version match was requested.

``settings``
  Search settings active for the search.

  ``required``
    The requirement request of the search. One of ``optional``,
    ``optional_explicit``, ``required_explicit``,
    ``required_from_package_variable``, or ``required_from_find_variable``.

  ``quiet``
    A boolean indicating whether the search is ``QUIET`` or not.

  ``global``
    A boolean indicating whether the ``GLOBAL`` keyword has been provided or
    not.

  ``policy_scope``
    A boolean indicating whether the ``NO_POLICY_SCOPE`` keyword has been
    provided or not.

  ``bypass_provider``
    A boolean indicating whether the ``BYPASS_PROVIDER`` keyword has been
    provided or not.

  ``hints``
    An array of paths provided as ``HINTS``.

  ``names``
    An array of package names to use when searching, provided by ``NAMES``.

  ``search_paths``
    An array of paths to search, provided by ``PATHS``.

  ``path_suffixes``
    An array of suffixes to use when searching, provided by ``PATH_SUFFIXES``.

  ``registry_view``
    The ``REGISTRY_VIEW`` requested for the search.

  ``paths``
    Path settings active for the search.

    ``CMAKE_FIND_USE_CMAKE_PATH``
      A boolean indicating whether or not CMake-specific cache variables are
      used when searching. See :variable:`CMAKE_FIND_USE_CMAKE_PATH`.

    ``CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH``
      A boolean indicating whether or not CMake-specific environment variables
      are used when searching. See
      :variable:`CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH`.

    ``CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH``
      A boolean indicating whether or not platform-specific environment
      variables are used when searching. See
      :variable:`CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH`.

    ``CMAKE_FIND_USE_CMAKE_SYSTEM_PATH``
      A boolean indicating whether or not platform-specific CMake variables are
      used when searching. See :variable:`CMAKE_FIND_USE_CMAKE_SYSTEM_PATH`.

    ``CMAKE_FIND_USE_INSTALL_PREFIX``
      A boolean indicating whether or not the install prefix is used when
      searching. See :variable:`CMAKE_FIND_USE_INSTALL_PREFIX`.

    ``CMAKE_FIND_USE_CMAKE_PACKAGE_REGISTRY``
      A boolean indicating whether or not to search the CMake package registry
      for the package. See :variable:`CMAKE_FIND_USE_PACKAGE_REGISTRY`.

    ``CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY``
      A boolean indicating whether or not to search the system CMake package
      registry for the package. See
      :variable:`CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY`.

    ``CMAKE_FIND_ROOT_PATH_MODE``
      A string indicating the root path mode in effect as selected by the
      ``CMAKE_FIND_ROOT_PATH_BOTH``, ``ONLY_CMAKE_FIND_ROOT_PATH``, and
      ``NO_CMAKE_FIND_ROOT_PATH`` arguments.

``candidates``
  An array of rejected candidate paths. Each element contains the following
  keys:

  ``path``
    The path to the considered file. In the case of a dependency provider, the
    value is in the form of ``dependency_provider::<COMMAND_NAME>``.

  ``mode``
    The mode which found the file. One of ``module``, ``cps``, ``cmake``, or
    ``provider``.

  ``reason``
    The reason the path was rejected. One of ``insufficient_version``,
    ``no_exist``, ``ignored``, ``no_config_file``, or ``not_found``.

  ``message``
    If present, a string describing why the package is considered as not
    found.

``found``
  If the package has been found, information on the found file. If it is not
  found, this is ``null``. Keys available:

  ``path``
    The path to the module or configuration that found the package. In the
    case of a dependency provider, the value is in the form of
    ``dependency_provider::<COMMAND_NAME>``.

  ``mode``
    The mode that considered the path. One of ``module``, ``cps``, ``cmake``,
    or ``provider``.

  ``version``
    The reported version of the package.

``search_context``
  A mapping of variable names to search paths specified by them (either a
  string or an array of strings depending on the variable). Environment
  variables are wrapped with ``ENV{`` and ``}``, otherwise CMake variables are
  used. Only variables with any paths specified are used.

  ``package_stack``
    An array of objects with paths which come from the stack of paths made
    available by :command:`find_package` calls.

    ``package_paths``
      The paths made available by :command:`find_package` commands in the call
      stack.
