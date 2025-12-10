INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
------------------------------------

List of public system include directories for a library.

Targets may populate this property to publish the include directories
which contain system headers, and therefore should not result in
compiler warnings.  Additionally, system include directories are searched
after normal include directories regardless of the order specified.

When the :command:`target_include_directories` command is given the
``SYSTEM`` keyword, it populates this property with values provided after the
``PUBLIC`` and ``INTERFACE`` keywords.

Projects may also get and set the property directly, but must be aware that
adding directories to this property does not make those directories used
during compilation.  Adding directories to this property marks directories
as system directories which otherwise would be used in a non-system manner.
This can appear similar to duplication, so prefer the high-level
:command:`target_include_directories` command with the ``SYSTEM`` keyword
and avoid setting the property directly.

When target dependencies are specified using :command:`target_link_libraries`,
CMake will read this property from all target dependencies to mark the
same include directories as containing system headers.

Contents of ``INTERFACE_SYSTEM_INCLUDE_DIRECTORIES`` may use "generator
expressions" with the syntax ``$<...>``.  See the
:manual:`cmake-generator-expressions(7)` manual for available expressions.
See the :manual:`cmake-buildsystem(7)` manual for more on defining
buildsystem properties.
