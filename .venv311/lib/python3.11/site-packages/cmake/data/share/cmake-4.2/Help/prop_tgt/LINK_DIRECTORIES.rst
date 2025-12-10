LINK_DIRECTORIES
----------------

.. versionadded:: 3.13

List of directories to use for the link step of shared library, module
and executable targets.

This property holds a :ref:`semicolon-separated list <CMake Language Lists>` of directories
specified so far for its target.  Use the :command:`target_link_directories`
command to append more search directories.

This property is initialized by the :prop_dir:`LINK_DIRECTORIES` directory
property when a target is created, and is used by the generators to set
the search directories for the linker.

Contents of ``LINK_DIRECTORIES`` may use "generator expressions" with the
syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)` manual
for available expressions.  See the :manual:`cmake-buildsystem(7)` manual
for more on defining buildsystem properties.
