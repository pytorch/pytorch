INCLUDE_DIRECTORIES
-------------------

.. versionadded:: 3.11

List of preprocessor include file search directories.

This property holds a :ref:`semicolon-separated list <CMake Language Lists>` of paths
and will be added to the list of include directories when this
source file builds. These directories will take precedence over directories
defined at target level except for :generator:`Xcode` generator due to technical
limitations.

Relative paths should not be added to this property directly.

Contents of ``INCLUDE_DIRECTORIES`` may use "generator expressions" with
the syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)` manual
for available expressions.  However, :generator:`Xcode` does not support
per-config per-source settings, so expressions that depend on the build
configuration are not allowed with that generator.
