Ninja Multi-Config
------------------

.. versionadded:: 3.17

Generates multiple ``build-<Config>.ninja`` files.

This generator is very much like the :generator:`Ninja` generator, but with
some key differences. Only these differences will be discussed in this
document.

Unlike the :generator:`Ninja` generator, ``Ninja Multi-Config`` generates
multiple configurations at once with :variable:`CMAKE_CONFIGURATION_TYPES`
instead of only one configuration with :variable:`CMAKE_BUILD_TYPE`. One
``build-<Config>.ninja`` file will be generated for each of these
configurations (with ``<Config>`` being the configuration name.) These files
are intended to be run with ``ninja -f build-<Config>.ninja``. A
``build.ninja`` file is also generated, using the configuration from either
:variable:`CMAKE_DEFAULT_BUILD_TYPE` or the first item from
:variable:`CMAKE_CONFIGURATION_TYPES`.

``cmake --build . --config <Config>`` will always use ``build-<Config>.ninja``
to build. If no :option:`--config <cmake--build --config>` argument is
specified, :option:`cmake --build . <cmake --build>` will use ``build.ninja``.

Each ``build-<Config>.ninja`` file contains ``<target>`` targets as well as
``<target>:<Config>`` targets, where ``<Config>`` is the same as the
configuration specified in ``build-<Config>.ninja`` Additionally, if
cross-config mode is enabled, ``build-<Config>.ninja`` may contain
``<target>:<OtherConfig>`` targets, where ``<OtherConfig>`` is a cross-config,
as well as ``<target>:all``, which builds the target in all cross-configs. See
below for how to enable cross-config mode.

The ``Ninja Multi-Config`` generator recognizes the following variables:

:variable:`CMAKE_CONFIGURATION_TYPES`
  Specifies the total set of configurations to build. Unlike with other
  multi-config generators, this variable has a value of
  ``Debug;Release;RelWithDebInfo`` by default.

:variable:`CMAKE_CROSS_CONFIGS`
  Specifies a :ref:`semicolon-separated list <CMake Language Lists>` of
  configurations available from all ``build-<Config>.ninja`` files.

:variable:`CMAKE_DEFAULT_BUILD_TYPE`
  Specifies the configuration to use by default in a ``build.ninja`` file.

:variable:`CMAKE_DEFAULT_CONFIGS`
  Specifies a :ref:`semicolon-separated list <CMake Language Lists>` of
  configurations to build for a target in ``build.ninja``
  if no ``:<Config>`` suffix is specified.

Consider the following example:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.16)
  project(MultiConfigNinja C)

  add_executable(generator generator.c)
  add_custom_command(OUTPUT generated.c COMMAND generator generated.c)
  add_library(generated ${CMAKE_BINARY_DIR}/generated.c)

Now assume you configure the project with ``Ninja Multi-Config`` and run one of
the following commands:

.. code-block:: shell

  ninja -f build-Debug.ninja generated
  # OR
  cmake --build . --config Debug --target generated

This would build the ``Debug`` configuration of ``generator``, which would be
used to generate ``generated.c``, which would be used to build the ``Debug``
configuration of ``generated``.

But if :variable:`CMAKE_CROSS_CONFIGS` is set to ``all``, and you run the
following instead:

.. code-block:: shell

  ninja -f build-Release.ninja generated:Debug
  # OR
  cmake --build . --config Release --target generated:Debug

This would build the ``Release`` configuration of ``generator``, which would be
used to generate ``generated.c``, which would be used to build the ``Debug``
configuration of ``generated``. This is useful for running a release-optimized
version of a generator utility while still building the debug version of the
targets built with the generated code.

Custom Commands
^^^^^^^^^^^^^^^

.. versionadded:: 3.20

The ``Ninja Multi-Config`` generator adds extra capabilities to
:command:`add_custom_command` and :command:`add_custom_target` through its
cross-config mode. The ``COMMAND``, ``DEPENDS``, and ``WORKING_DIRECTORY``
arguments can be evaluated in the context of either the "command config" (the
"native" configuration of the ``build-<Config>.ninja`` file in use) or the
"output config" (the configuration used to evaluate the ``OUTPUT`` and
``BYPRODUCTS``).

If either ``OUTPUT`` or ``BYPRODUCTS`` names a path that is common to
more than one configuration (e.g. it does not use any generator expressions),
all arguments are evaluated in the command config by default.
If all ``OUTPUT`` and ``BYPRODUCTS`` paths are unique to each configuration
(e.g. by using the :genex:`$<CONFIG>` generator expression), the first argument of
``COMMAND`` is still evaluated in the command config by default, while all
subsequent arguments, as well as the arguments to ``DEPENDS`` and
``WORKING_DIRECTORY``, are evaluated in the output config. These defaults can
be overridden with the :genex:`$<OUTPUT_CONFIG:...>` and :genex:`$<COMMAND_CONFIG:...>`
generator-expressions. Note that if a target is specified by its name in
``DEPENDS``, or as the first argument of ``COMMAND``, it is always evaluated
in the command config, even if it is wrapped in :genex:`$<OUTPUT_CONFIG:...>`
(because its plain name is not a generator expression).

As an example, consider the following:

.. code-block:: cmake

  add_custom_command(
    OUTPUT "$<CONFIG>.txt"
    COMMAND
      generator "$<CONFIG>.txt"
                "$<OUTPUT_CONFIG:$<CONFIG>>"
                "$<COMMAND_CONFIG:$<CONFIG>>"
    DEPENDS
      tgt1
      "$<TARGET_FILE:tgt2>"
      "$<OUTPUT_CONFIG:$<TARGET_FILE:tgt3>>"
      "$<COMMAND_CONFIG:$<TARGET_FILE:tgt4>>"
    )

Assume that ``generator``, ``tgt1``, ``tgt2``, ``tgt3``, and ``tgt4`` are all
executable targets, and assume that ``$<CONFIG>.txt`` is built in the ``Debug``
output config using the ``Release`` command config. The ``Release`` build of
the ``generator`` target is called with ``Debug.txt Debug Release`` as
arguments. The command depends on the ``Release`` builds of ``tgt1`` and
``tgt4``, and the ``Debug`` builds of ``tgt2`` and ``tgt3``.

``PRE_BUILD``, ``PRE_LINK``, and ``POST_BUILD`` custom commands for targets
only get run in their "native" configuration (the ``Release`` configuration in
the ``build-Release.ninja`` file) unless they have no ``BYPRODUCTS`` or their
``BYPRODUCTS`` are unique per config. Consider the following example:

.. code-block:: cmake

  add_executable(exe main.c)
  add_custom_command(
    TARGET exe
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E echo "Running no-byproduct command"
    )
  add_custom_command(
    TARGET exe
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E echo
      "Running separate-byproduct command for $<CONFIG>"
    BYPRODUCTS $<CONFIG>.txt
    )
  add_custom_command(
    TARGET exe
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E echo
      "Running common-byproduct command for $<CONFIG>"
    BYPRODUCTS exe.txt
    )

In this example, if you build ``exe:Debug`` in ``build-Release.ninja``, the
first and second custom commands get run, since their byproducts are unique
per-config, but the last custom command does not. However, if you build
``exe:Release`` in ``build-Release.ninja``, all three custom commands get run.
