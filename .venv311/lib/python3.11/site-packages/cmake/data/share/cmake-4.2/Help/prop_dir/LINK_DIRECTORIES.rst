LINK_DIRECTORIES
----------------

List of linker search directories.

This property holds a :ref:`semicolon-separated list <CMake Language Lists>` of directories
and is typically populated using the :command:`link_directories` command.
It gets its initial value from its parent directory, if it has one.

The directory property is used to initialize the :prop_tgt:`LINK_DIRECTORIES`
target property when a target is created.  That target property is used
by the generators to set the library search directories for the linker.

Contents of ``LINK_DIRECTORIES`` may use "generator expressions" with
the syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)`
manual for available expressions.  See the :manual:`cmake-buildsystem(7)`
manual for more on defining buildsystem properties.
