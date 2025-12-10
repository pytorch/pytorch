try_run
-------

.. only:: html

   .. contents::

Try compiling and then running some code.

Try Compiling and Running Source Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cmake

  try_run(<runResultVar> <compileResultVar>
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
          [COMPILE_OUTPUT_VARIABLE <var>]
          [COPY_FILE <fileName> [COPY_FILE_ERROR <var>]]
          [<LANG>_STANDARD <std>]
          [<LANG>_STANDARD_REQUIRED <bool>]
          [<LANG>_EXTENSIONS <bool>]
          [RUN_OUTPUT_VARIABLE <var>]
          [RUN_OUTPUT_STDOUT_VARIABLE <var>]
          [RUN_OUTPUT_STDERR_VARIABLE <var>]
          [WORKING_DIRECTORY <var>]
          [ARGS <args>...]
          )

.. versionadded:: 3.25

Try building an executable from one or more source files.  Build success
returns boolean ``true`` and build failure returns boolean ``false`` in
``<compileResultVar>`` (cached unless ``NO_CACHE`` is specified).
If the build succeeds, this runs the executable and stores the exit code
in ``<runResultVar>`` (cached unless ``NO_CACHE`` is specified).
If the executable was built, but failed to run, then ``<runResultVar>``
will be set to ``FAILED_TO_RUN``.  See command :command:`try_compile` for
documentation of options common to both commands, and for information on
how the test project is constructed to build the source file.

One or more source files must be provided. Additionally, one of ``SOURCES``
and/or ``SOURCE_FROM_*`` must precede other keywords.

.. versionadded:: 3.26
  This command records a
  :ref:`configure-log try_run event <try_run configure-log event>`
  if the ``NO_LOG`` option is not specified.

This command supports an alternate signature for CMake older than 3.25.
The signature above is recommended for clarity.

.. code-block:: cmake

  try_run(<runResultVar> <compileResultVar>
          <bindir> <srcfile|SOURCES srcfile...>
          [CMAKE_FLAGS <flags>...]
          [COMPILE_DEFINITIONS <defs>...]
          [LINK_OPTIONS <options>...]
          [LINK_LIBRARIES <libs>...]
          [LINKER_LANGUAGE <lang>]
          [COMPILE_OUTPUT_VARIABLE <var>]
          [COPY_FILE <fileName> [COPY_FILE_ERROR <var>]]
          [<LANG>_STANDARD <std>]
          [<LANG>_STANDARD_REQUIRED <bool>]
          [<LANG>_EXTENSIONS <bool>]
          [RUN_OUTPUT_VARIABLE <var>]
          [OUTPUT_VARIABLE <var>]
          [WORKING_DIRECTORY <var>]
          [ARGS <args>...]
          )

.. _`try_run Options`:

Options
^^^^^^^

The options specific to ``try_run`` are:

``COMPILE_OUTPUT_VARIABLE <var>``
  Report the compile step build output in a given variable.

``OUTPUT_VARIABLE <var>``
  Report the compile build output and the output from running the executable
  in the given variable.  This option exists for legacy reasons and is only
  supported by the old ``try_run`` signature.
  Prefer ``COMPILE_OUTPUT_VARIABLE`` and ``RUN_OUTPUT_VARIABLE`` instead.

``RUN_OUTPUT_VARIABLE <var>``
  Report the output from running the executable in a given variable.

``RUN_OUTPUT_STDOUT_VARIABLE <var>``
  .. versionadded:: 3.25

  Report the output of stdout from running the executable in a given variable.

``RUN_OUTPUT_STDERR_VARIABLE <var>``
  .. versionadded:: 3.25

  Report the output of stderr from running the executable in a given variable.

``WORKING_DIRECTORY <var>``
  .. versionadded:: 3.20

  Run the executable in the given directory. If no ``WORKING_DIRECTORY`` is
  specified, the executable will run in ``<bindir>`` or the current build
  directory.

``ARGS <args>...``
  Additional arguments to pass to the executable when running it.

Other Behavior Settings
^^^^^^^^^^^^^^^^^^^^^^^

Set variable :variable:`CMAKE_TRY_COMPILE_CONFIGURATION` to choose a build
configuration:

* For multi-config generators, this selects which configuration to build.

* For single-config generators, this sets :variable:`CMAKE_BUILD_TYPE` in
  the test project.

Behavior when Cross Compiling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.3
  Use ``CMAKE_CROSSCOMPILING_EMULATOR`` when running cross-compiled binaries.

When cross compiling, the executable compiled in the first step
usually cannot be run on the build host.  The ``try_run`` command checks
the :variable:`CMAKE_CROSSCOMPILING` variable to detect whether CMake is in
cross-compiling mode.  If that is the case, it will still try to compile
the executable, but it will not try to run the executable unless the
:variable:`CMAKE_CROSSCOMPILING_EMULATOR` variable is set.  Instead it
will create cache variables which must be filled by the user or by
presetting them in some CMake script file to the values the executable
would have produced if it had been run on its actual target platform.
These cache entries are:

``<runResultVar>``
  Exit code if the executable were to be run on the target platform.

``<runResultVar>__TRYRUN_OUTPUT``
  Output from stdout and stderr if the executable were to be run on
  the target platform.  This is created only if the
  ``RUN_OUTPUT_VARIABLE`` or ``OUTPUT_VARIABLE`` option was used.

``<runResultVar>__TRYRUN_OUTPUT_STDOUT``
  .. versionadded:: 3.25

  Output from stdout if the executable were to be run on the target
  platform.  This is created only if the ``RUN_OUTPUT_STDOUT_VARIABLE``
  or ``RUN_OUTPUT_STDERR_VARIABLE`` option was used.

``<runResultVar>__TRYRUN_OUTPUT_STDERR``
  .. versionadded:: 3.25

  Output from stderr if the executable were to be run on the target
  platform.  This is created only if the ``RUN_OUTPUT_STDOUT_VARIABLE``
  or ``RUN_OUTPUT_STDERR_VARIABLE`` option was used.

In order to make cross compiling your project easier, use ``try_run``
only if really required.  If you use ``try_run``, use the
``RUN_OUTPUT_STDOUT_VARIABLE``, ``RUN_OUTPUT_STDERR_VARIABLE``,
``RUN_OUTPUT_VARIABLE`` or ``OUTPUT_VARIABLE`` options only if really
required.  Using them will require that when cross-compiling, the cache
variables will have to be set manually to the output of the executable.
You can also "guard" the calls to ``try_run`` with an :command:`if`
block checking the :variable:`CMAKE_CROSSCOMPILING` variable and
provide an easy-to-preset alternative for this case.
