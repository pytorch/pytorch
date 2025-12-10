COMPILE_DEFINITIONS
-------------------

Preprocessor definitions for compiling a source file.

The ``COMPILE_DEFINITIONS`` property may be set to a semicolon-separated
list of preprocessor definitions using the syntax ``VAR`` or ``VAR=value``.
Function-style definitions are not supported.  CMake will
automatically escape the value correctly for the native build system
(note that CMake language syntax may require escapes to specify some
values).  This property may be set on a per-configuration basis using
the name ``COMPILE_DEFINITIONS_<CONFIG>`` where ``<CONFIG>`` is an upper-case
name (ex.  ``COMPILE_DEFINITIONS_DEBUG``).

CMake will automatically drop some definitions that are not supported
by the native build tool.  :generator:`Xcode` does not support per-configuration
definitions on source files.

.. versionadded:: 3.26
  Any leading ``-D`` on an item will be removed.

.. include:: /include/COMPILE_DEFINITIONS_DISCLAIMER.rst

Contents of ``COMPILE_DEFINITIONS`` may use :manual:`cmake-generator-expressions(7)`
with the syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)`
manual for available expressions.  However, :generator:`Xcode`
does not support per-config per-source settings, so expressions
that depend on the build configuration are not allowed with that
generator.

Prefer using generator expressions in :prop_sf:`!COMPILE_DEFINITIONS` over the
deprecated :prop_sf:`COMPILE_DEFINITIONS_<CONFIG>` property.
