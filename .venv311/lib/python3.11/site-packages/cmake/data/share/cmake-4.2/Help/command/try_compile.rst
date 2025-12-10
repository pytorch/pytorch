try_compile
-----------

.. only:: html

   .. contents::

Try building some code.

.. _`Try Compiling Whole Projects`:

Try Compiling Whole Projects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cmake

  try_compile(<compileResultVar> PROJECT <projectName>
              SOURCE_DIR <srcdir>
              [BINARY_DIR <bindir>]
              [TARGET <targetName>]
              [LOG_DESCRIPTION <text>]
              [NO_CACHE]
              [NO_LOG]
              [CMAKE_FLAGS <flags>...]
              [OUTPUT_VARIABLE <var>])

.. versionadded:: 3.25

Try building a project.  Build success returns ``TRUE`` and build failure
returns ``FALSE`` in ``<compileResultVar>``.

In this form, ``<srcdir>`` should contain a complete CMake project with a
``CMakeLists.txt`` file and all sources.  The ``<bindir>`` and ``<srcdir>``
will not be deleted after this command is run.  Specify ``<targetName>`` to
build a specific target instead of the ``all`` or ``ALL_BUILD`` target.  See
below for the meaning of other options.

.. versionchanged:: 3.24
  CMake variables describing platform settings, and those listed by the
  :variable:`CMAKE_TRY_COMPILE_PLATFORM_VARIABLES` variable, are propagated
  into the project's build configuration.  See policy :policy:`CMP0137`.
  Previously this was only done by the
  :ref:`source file <Try Compiling Source Files>` signature.

.. versionadded:: 3.26
  This command records a
  :ref:`configure-log try_compile event <try_compile configure-log event>`
  if the ``NO_LOG`` option is not specified.

.. versionadded:: 3.30
  If the :prop_gbl:`PROPAGATE_TOP_LEVEL_INCLUDES_TO_TRY_COMPILE` global
  property is set to true, :variable:`CMAKE_PROJECT_TOP_LEVEL_INCLUDES` is
  propagated into the project's build configuration.

This command supports an alternate signature for CMake older than 3.25.
The signature above is recommended for clarity.

.. code-block:: cmake

  try_compile(<compileResultVar> <bindir> <srcdir>
              <projectName> [<targetName>]
              [CMAKE_FLAGS <flags>...]
              [OUTPUT_VARIABLE <var>])

.. _`Try Compiling Source Files`:

Try Compiling Source Files
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cmake

  try_compile(<compileResultVar>
              [SOURCES_TYPE <type>]
              <SOURCES <srcfile...>                 |
               SOURCE_FROM_CONTENT <name> <content> |
               SOURCE_FROM_VAR <name> <var>         |
               SOURCE_FROM_FILE <name> <path>       >...
              [LOG_DESCRIPTION <text>]
              [NO_CACHE]
              [NO_LOG]
              [CMAKE_FLAGS <flags>...]
              [COMPILE_DEFINITIONS <defs>...]
              [LINK_OPTIONS <options>...]
              [LINK_LIBRARIES <libs>...]
              [LINKER_LANGUAGE <lang>]
              [OUTPUT_VARIABLE <var>]
              [COPY_FILE <fileName> [COPY_FILE_ERROR <var>]]
              [<LANG>_STANDARD <std>]
              [<LANG>_STANDARD_REQUIRED <bool>]
              [<LANG>_EXTENSIONS <bool>]
              )

.. versionadded:: 3.25

Try building an executable or static library from one or more source files.
The binary type is determined by variable
:variable:`CMAKE_TRY_COMPILE_TARGET_TYPE`.
Build success returns boolean ``true`` and build failure returns boolean
``false`` in ``<compileResultVar>`` (cached unless ``NO_CACHE`` is specified).

In this form, one or more source files must be provided. Additionally, one of
``SOURCES`` and/or ``SOURCE_FROM_*`` must precede other keywords.

If :variable:`CMAKE_TRY_COMPILE_TARGET_TYPE` is unset or is set to
``EXECUTABLE``, the sources must include a definition for ``main`` and CMake
will create a ``CMakeLists.txt`` file to build the source(s) as an executable.
If :variable:`CMAKE_TRY_COMPILE_TARGET_TYPE` is set to ``STATIC_LIBRARY``,
a static library will be built instead and no definition for ``main`` is
required.  For an executable, the generated ``CMakeLists.txt`` file would
contain something like the following:

.. code-block:: cmake

  add_definitions(<expanded COMPILE_DEFINITIONS from caller>)
  include_directories(${INCLUDE_DIRECTORIES})
  link_directories(${LINK_DIRECTORIES})
  add_executable(cmTryCompileExec <srcfile>...)
  target_link_options(cmTryCompileExec PRIVATE <LINK_OPTIONS from caller>)
  target_link_libraries(cmTryCompileExec ${LINK_LIBRARIES})

CMake automatically generates, for each ``try_compile`` operation, a
unique directory under ``${CMAKE_BINARY_DIR}/CMakeFiles/CMakeScratch``
with an unspecified name.  These directories are cleaned automatically unless
:option:`--debug-trycompile <cmake --debug-trycompile>` is passed to :program:`cmake`.
Such directories from previous runs are also unconditionally cleaned at the
beginning of any :program:`cmake` execution.

This command supports an alternate signature for CMake older than 3.25.
The signature above is recommended for clarity.

.. code-block:: cmake

  try_compile(<compileResultVar> <bindir> <srcfile|SOURCES srcfile...>
              [CMAKE_FLAGS <flags>...]
              [COMPILE_DEFINITIONS <defs>...]
              [LINK_OPTIONS <options>...]
              [LINK_LIBRARIES <libs>...]
              [OUTPUT_VARIABLE <var>]
              [COPY_FILE <fileName> [COPY_FILE_ERROR <var>]]
              [<LANG>_STANDARD <std>]
              [<LANG>_STANDARD_REQUIRED <bool>]
              [<LANG>_EXTENSIONS <bool>]
              )

In this version, ``try_compile`` will use ``<bindir>/CMakeFiles/CMakeTmp`` for
its operation, and all such files will be cleaned automatically.
For debugging, :option:`--debug-trycompile <cmake --debug-trycompile>` can be
passed to :program:`cmake` to avoid this clean.  However, multiple sequential
``try_compile`` operations, if given the same ``<bindir>``, will reuse this
single output directory, such that you can only debug one such ``try_compile``
call at a time.  Use of the newer signature is recommended to simplify
debugging of multiple ``try_compile`` operations.

.. _`try_compile Options`:

Options
^^^^^^^

The options for the above signatures are:

``CMAKE_FLAGS <flags>...``
  Specify flags of the form :option:`-DVAR:TYPE=VALUE <cmake -D>` to be passed
  to the :manual:`cmake(1)` command-line used to drive the test build.
  The above example shows how values for variables
  ``COMPILE_DEFINITIONS``, ``INCLUDE_DIRECTORIES``, ``LINK_DIRECTORIES``,
  ``LINK_LIBRARIES``, and ``LINK_OPTIONS`` are used. Compiler options
  can be passed in like ``CMAKE_FLAGS -DCOMPILE_DEFINITIONS=-Werror``.

``COMPILE_DEFINITIONS <defs>...``
  Specify ``-Ddefinition`` arguments to pass to :command:`add_definitions`
  in the generated test project.

``COPY_FILE <fileName>``
  Copy the built executable or static library to the given ``<fileName>``.

``COPY_FILE_ERROR <var>``
  Use after ``COPY_FILE`` to capture into variable ``<var>`` any error
  message encountered while trying to copy the file.

``LINK_LIBRARIES <libs>...``
  Specify libraries to be linked in the generated project.
  The list of libraries may refer to system libraries and to
  :ref:`Imported Targets <Imported Targets>` from the calling project.

  If this option is specified, any ``-DLINK_LIBRARIES=...`` value
  given to the ``CMAKE_FLAGS`` option will be ignored.

  .. versionadded:: 3.29
    Alias targets to imported libraries are also supported.

``LINK_OPTIONS <options>...``
  .. versionadded:: 3.14

  Specify link step options to pass to :command:`target_link_options` or to
  set the :prop_tgt:`STATIC_LIBRARY_OPTIONS` target property in the generated
  project, depending on the :variable:`CMAKE_TRY_COMPILE_TARGET_TYPE` variable.

``LINKER_LANGUAGE <lang>``
  .. versionadded:: 3.29

  Specify the :prop_tgt:`LINKER_LANGUAGE` target property of the generated
  project.  When using multiple source files with different languages, set
  this to the language of the source file containing the program entry point,
  e.g., ``main``.

``LOG_DESCRIPTION <text>``
  .. versionadded:: 3.26

  Specify a non-empty text description of the purpose of the check.
  This is recorded in the :manual:`cmake-configure-log(7)` entry.

``NO_CACHE``
  .. versionadded:: 3.25

  ``<compileResultVar>`` will be stored in a normal variable rather than a
  cache entry.

  ``<compileResultVar>`` is normally cached so that a simple pattern can be used
  to avoid repeating the test on subsequent executions of CMake:

  .. code-block:: cmake

    if(NOT DEFINED RESULTVAR)
      # ...(check-specific setup code)...
      try_compile(RESULTVAR ...)
      # ...(check-specific logging and cleanup code)...
    endif()

  If the guard variable and result variable are not the same (for example, if
  the test is part of a larger inspection), ``NO_CACHE`` may be useful to avoid
  leaking the intermediate result variable into the cache.

``NO_LOG``
  .. versionadded:: 3.26

  Do not record a :manual:`cmake-configure-log(7)` entry for this call.

``OUTPUT_VARIABLE <var>``
  Store the output from the build process in the given variable.

``SOURCE_FROM_CONTENT <name> <content>``
  .. versionadded:: 3.25

  Write ``<content>`` to a file named ``<name>`` in the operation directory.
  This can be used to bypass the need to separately write a source file when
  the contents of the file are dynamically specified. The specified ``<name>``
  is not allowed to contain path components.

  ``SOURCE_FROM_CONTENT`` may be specified multiple times.

``SOURCE_FROM_FILE <name> <path>``
  .. versionadded:: 3.25

  Copy ``<path>`` to a file named ``<name>`` in the operation directory. This
  can be used to consolidate files into the operation directory, which may be
  useful if a source which already exists (i.e. as a stand-alone file in a
  project's source repository) needs to refer to other file(s) created by
  ``SOURCE_FROM_*``. (Otherwise, ``SOURCES`` is usually more convenient.) The
  specified ``<name>`` is not allowed to contain path components.

``SOURCE_FROM_VAR <name> <var>``
  .. versionadded:: 3.25

  Write the contents of ``<var>`` to a file named ``<name>`` in the operation
  directory. This is the same as ``SOURCE_FROM_CONTENT``, but takes the
  contents from the specified CMake variable, rather than directly, which may
  be useful when passing arguments through a function which wraps
  ``try_compile``. The specified ``<name>`` is not allowed to contain path
  components.

  ``SOURCE_FROM_VAR`` may be specified multiple times.

``SOURCES_TYPE <type>``
  .. versionadded:: 3.28

  Sources may be classified using the ``SOURCES_TYPE`` argument. Once
  specified, all subsequent sources specified will be treated as that type
  until another ``SOURCES_TYPE`` is given. Available types are:

  ``NORMAL``
    Sources are not added to any ``FILE_SET`` in the generated project.

  ``CXX_MODULE``
    .. versionadded:: 3.28

    Sources are added to a ``FILE_SET`` of type ``CXX_MODULES`` in the
    generated project.

  The default type of sources is ``NORMAL``.

``<LANG>_STANDARD <std>``
  .. versionadded:: 3.8

  Specify the :prop_tgt:`C_STANDARD`, :prop_tgt:`CXX_STANDARD`,
  :prop_tgt:`OBJC_STANDARD`, :prop_tgt:`OBJCXX_STANDARD`,
  or :prop_tgt:`CUDA_STANDARD` target property of the generated project.

``<LANG>_STANDARD_REQUIRED <bool>``
  .. versionadded:: 3.8

  Specify the :prop_tgt:`C_STANDARD_REQUIRED`,
  :prop_tgt:`CXX_STANDARD_REQUIRED`, :prop_tgt:`OBJC_STANDARD_REQUIRED`,
  :prop_tgt:`OBJCXX_STANDARD_REQUIRED`,or :prop_tgt:`CUDA_STANDARD_REQUIRED`
  target property of the generated project.

``<LANG>_EXTENSIONS <bool>``
  .. versionadded:: 3.8

  Specify the :prop_tgt:`C_EXTENSIONS`, :prop_tgt:`CXX_EXTENSIONS`,
  :prop_tgt:`OBJC_EXTENSIONS`, :prop_tgt:`OBJCXX_EXTENSIONS`,
  or :prop_tgt:`CUDA_EXTENSIONS` target property of the generated project.

Other Behavior Settings
^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.4
  If set, the following variables are passed in to the generated
  try_compile CMakeLists.txt to initialize compile target properties with
  default values:

  * :variable:`CMAKE_CUDA_RUNTIME_LIBRARY`
  * :variable:`CMAKE_ENABLE_EXPORTS`
  * :variable:`CMAKE_EXE_LINKER_FLAGS`, unless using CMake versions
    prior to 4.0 without policy :policy:`CMP0056` set to ``NEW``
  * :variable:`CMAKE_LINK_SEARCH_START_STATIC`
  * :variable:`CMAKE_LINK_SEARCH_END_STATIC`
  * :variable:`CMAKE_MSVC_RUNTIME_LIBRARY`
  * :variable:`CMAKE_POSITION_INDEPENDENT_CODE`
  * :variable:`CMAKE_WATCOM_RUNTIME_LIBRARY`

.. versionchanged:: 3.14
  If :policy:`CMP0083` is set to ``NEW``, then in order to obtain correct
  behavior at link time, the ``check_pie_supported()`` command from the
  :module:`CheckPIESupported` module must be called before using the
  ``try_compile`` command.

Some policies are set automatically in the generated test project
as needed to honor the state of the calling project:

* :policy:`CMP0065` (in CMake versions prior to 4.0)
* :policy:`CMP0083`
* :policy:`CMP0091`
* :policy:`CMP0104`
* :policy:`CMP0123`
* :policy:`CMP0126`
* :policy:`CMP0128`
* :policy:`CMP0136`
* :policy:`CMP0141`
* :policy:`CMP0155`
* :policy:`CMP0157`
* :policy:`CMP0181`
* :policy:`CMP0184`

.. versionadded:: 4.0
  The current setting of :policy:`CMP0181` policy is propagated through to the
  generated test project.

Set variable :variable:`CMAKE_TRY_COMPILE_CONFIGURATION` to choose a build
configuration:

* For multi-config generators, this selects which configuration to build.

* For single-config generators, this sets :variable:`CMAKE_BUILD_TYPE` in
  the test project.

.. versionadded:: 3.6
  Set the :variable:`CMAKE_TRY_COMPILE_TARGET_TYPE` variable to specify
  the type of target used for the source file signature.

.. versionadded:: 3.6
  Set the :variable:`CMAKE_TRY_COMPILE_PLATFORM_VARIABLES` variable to specify
  variables that must be propagated into the test project.  This variable is
  meant for use only in toolchain files and is only honored by the
  ``try_compile()`` command for the source files form, not when given a whole
  project.

.. versionchanged:: 3.8
  If :policy:`CMP0067` is set to ``NEW``, or any of the ``<LANG>_STANDARD``,
  ``<LANG>_STANDARD_REQUIRED``, or ``<LANG>_EXTENSIONS`` options are used,
  then the language standard variables are honored:

  * :variable:`CMAKE_C_STANDARD`
  * :variable:`CMAKE_C_STANDARD_REQUIRED`
  * :variable:`CMAKE_C_EXTENSIONS`
  * :variable:`CMAKE_CXX_STANDARD`
  * :variable:`CMAKE_CXX_STANDARD_REQUIRED`
  * :variable:`CMAKE_CXX_EXTENSIONS`
  * :variable:`CMAKE_OBJC_STANDARD`
  * :variable:`CMAKE_OBJC_STANDARD_REQUIRED`
  * :variable:`CMAKE_OBJC_EXTENSIONS`
  * :variable:`CMAKE_OBJCXX_STANDARD`
  * :variable:`CMAKE_OBJCXX_STANDARD_REQUIRED`
  * :variable:`CMAKE_OBJCXX_EXTENSIONS`
  * :variable:`CMAKE_CUDA_STANDARD`
  * :variable:`CMAKE_CUDA_STANDARD_REQUIRED`
  * :variable:`CMAKE_CUDA_EXTENSIONS`

  Their values are used to set the corresponding target properties in
  the generated project (unless overridden by an explicit option).

.. versionchanged:: 3.14
  For the :generator:`Green Hills MULTI` generator, the GHS toolset and target
  system customization cache variables are also propagated into the test
  project.

.. versionadded:: 3.24
  The :variable:`CMAKE_TRY_COMPILE_NO_PLATFORM_VARIABLES` variable may be
  set to disable passing platform variables into the test project.

.. versionadded:: 3.25
  If :policy:`CMP0141` is set to ``NEW``, one can use
  :variable:`CMAKE_MSVC_DEBUG_INFORMATION_FORMAT` to specify the MSVC debug
  information format.

.. versionadded:: 3.30
  If the :prop_gbl:`PROPAGATE_TOP_LEVEL_INCLUDES_TO_TRY_COMPILE` global
  property is set to true, :variable:`CMAKE_PROJECT_TOP_LEVEL_INCLUDES` is
  propagated into the test project's build configuration when using the
  :ref:`whole-project signature <Try Compiling Whole Projects>`.

.. versionadded:: 4.0
  If :policy:`CMP0184` is set to ``NEW``, one can use
  :variable:`CMAKE_MSVC_RUNTIME_CHECKS` to specify the enabled MSVC runtime
  checks.

See Also
^^^^^^^^

* :command:`try_run`
