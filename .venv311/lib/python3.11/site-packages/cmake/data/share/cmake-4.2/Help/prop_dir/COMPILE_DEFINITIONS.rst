COMPILE_DEFINITIONS
-------------------

Preprocessor definitions for compiling a directory's sources.

This property specifies the list of options given so far to the
:command:`add_compile_definitions` (or :command:`add_definitions`) command.

The ``COMPILE_DEFINITIONS`` property may be set to a semicolon-separated
list of preprocessor definitions using the syntax ``VAR`` or ``VAR=value``.
Function-style definitions are not supported.  CMake will
automatically escape the value correctly for the native build system
(note that CMake language syntax may require escapes to specify some
values).

This property will be initialized in each directory by its value in the
directory's parent.

CMake will automatically drop some definitions that are not supported
by the native build tool.

.. versionadded:: 3.26
  Any leading ``-D`` on an item will be removed.

.. include:: /include/COMPILE_DEFINITIONS_DISCLAIMER.rst

Contents of ``COMPILE_DEFINITIONS`` may use "generator expressions" with
the syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)`
manual for available expressions.  See the :manual:`cmake-buildsystem(7)`
manual for more on defining buildsystem properties.

The corresponding :prop_dir:`COMPILE_DEFINITIONS_<CONFIG>` property may
be set to specify per-configuration definitions.  Generator expressions
should be preferred instead of setting the alternative property.
