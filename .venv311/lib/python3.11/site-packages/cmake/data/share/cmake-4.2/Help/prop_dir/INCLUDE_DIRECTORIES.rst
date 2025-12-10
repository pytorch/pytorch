INCLUDE_DIRECTORIES
-------------------

List of preprocessor include file search directories.

This property specifies the list of directories given so far to the
:command:`include_directories` command.

This property is used to populate the :prop_tgt:`INCLUDE_DIRECTORIES`
target property, which is used by the generators to set the include
directories for the compiler.

In addition to accepting values from that command, values may be set
directly on any directory using the :command:`set_property` command, and can be
set on the current directory using the :command:`set_directory_properties`
command.  A directory gets its initial value from its parent directory if it has
one.  The initial value of the :prop_tgt:`INCLUDE_DIRECTORIES` target property
comes from the value of this property.  Both directory and target property
values are adjusted by calls to the :command:`include_directories` command.
Calls to :command:`set_property` or :command:`set_directory_properties`,
however, will update the directory property value without updating target
property values.  Therefore direct property updates must be made before
calls to :command:`add_executable` or :command:`add_library` for targets
they are meant to affect.

The target property values are used by the generators to set the
include paths for the compiler.

Contents of ``INCLUDE_DIRECTORIES`` may use "generator expressions" with
the syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)`
manual for available expressions.  See the :manual:`cmake-buildsystem(7)`
manual for more on defining buildsystem properties.
