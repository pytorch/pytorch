target_compile_options
----------------------

Add compile options to a target.

.. code-block:: cmake

  target_compile_options(<target> [BEFORE]
    <INTERFACE|PUBLIC|PRIVATE> [items1...]
    [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])

Adds options to the :prop_tgt:`COMPILE_OPTIONS` or
:prop_tgt:`INTERFACE_COMPILE_OPTIONS` target properties. These options
are used when compiling the given ``<target>``, which must have been
created by a command such as :command:`add_executable` or
:command:`add_library` and must not be an :ref:`ALIAS target <Alias Targets>`.

.. note::

  These options are not used when linking the target.
  See the :command:`target_link_options` command for that.

Arguments
^^^^^^^^^

If ``BEFORE`` is specified, the content will be prepended to the property
instead of being appended.  See policy :policy:`CMP0101` which affects
whether ``BEFORE`` will be ignored in certain cases.

The ``INTERFACE``, ``PUBLIC`` and ``PRIVATE`` keywords are required to
specify the :ref:`scope <Target Command Scope>` of the following arguments.
``PRIVATE`` and ``PUBLIC`` items will populate the :prop_tgt:`COMPILE_OPTIONS`
property of ``<target>``.  ``PUBLIC`` and ``INTERFACE`` items will populate the
:prop_tgt:`INTERFACE_COMPILE_OPTIONS` property of ``<target>``.
The following arguments specify compile options.  Repeated calls for the same
``<target>`` append items in the order called.

.. versionadded:: 3.11
  Allow setting ``INTERFACE`` items on :ref:`IMPORTED targets <Imported Targets>`.

.. |command_name| replace:: ``target_compile_options``
.. include:: include/GENEX_NOTE.rst

.. include:: include/OPTIONS_SHELL.rst

See Also
^^^^^^^^

* This command can be used to add any options. However, for adding
  preprocessor definitions and include directories it is recommended
  to use the more specific commands :command:`target_compile_definitions`
  and :command:`target_include_directories`.

* For directory-wide settings, there is the command :command:`add_compile_options`.

* For file-specific settings, there is the source file property :prop_sf:`COMPILE_OPTIONS`.

* This command adds compile options for all languages in a target.
  Use the :genex:`COMPILE_LANGUAGE` generator expression to specify
  per-language compile options.

* :command:`target_compile_features`
* :command:`target_link_libraries`
* :command:`target_link_directories`
* :command:`target_link_options`
* :command:`target_precompile_headers`
* :command:`target_sources`

* :variable:`CMAKE_<LANG>_FLAGS` and :variable:`CMAKE_<LANG>_FLAGS_<CONFIG>`
  add language-wide flags passed to all invocations of the compiler.
  This includes invocations that drive compiling and those that drive linking.

* The :module:`CheckCompilerFlag` module to check whether the compiler
  supports a given flag.
