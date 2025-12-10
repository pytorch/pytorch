LOCATION
--------

Read-only location of a target on disk.

For an imported target, this read-only property returns the value of
the ``LOCATION_<CONFIG>`` property for an unspecified configuration
``<CONFIG>`` provided by the target.

For a non-imported target, this property is provided for compatibility
with CMake 2.4 and below.  It was meant to get the location of an
executable target's output file for use in :command:`add_custom_command`.  The
path may contain a build-system-specific portion that is replaced at
build time with the configuration getting built (such as
``$(ConfigurationName)`` in VS).  In CMake 2.6 and above
:command:`add_custom_command` automatically recognizes a target name in its
``COMMAND`` and ``DEPENDS`` options and computes the target location.  In
CMake 2.8.4 and above :command:`add_custom_command` recognizes
:manual:`generator expressions <cmake-generator-expressions(7)>`
to refer to target locations anywhere in the command.
Therefore this property is not needed for creating custom commands.

Do not set properties that affect the location of a target after
reading this property.  These include properties whose names match
``(RUNTIME|LIBRARY|ARCHIVE)_OUTPUT_(NAME|DIRECTORY)(_<CONFIG>)?``,
``(IMPLIB_)?(PREFIX|SUFFIX)``, or "LINKER_LANGUAGE".  Failure to follow
this rule is not diagnosed and leaves the location of the target
undefined.
