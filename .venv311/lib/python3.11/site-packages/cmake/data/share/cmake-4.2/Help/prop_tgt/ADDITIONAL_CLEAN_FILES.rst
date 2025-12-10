ADDITIONAL_CLEAN_FILES
----------------------

.. versionadded:: 3.15

A :ref:`;-list <CMake Language Lists>` of files or directories that will be
removed as a part of the global ``clean`` target.  It can be used to specify
files and directories that are generated as part of building the target or
that are directly associated with the target in some way (e.g. created as a
result of running the target).

For custom targets, if such files can be captured as outputs or byproducts
instead, then that should be preferred over adding them to this property.
If an additional clean file is used by multiple targets or isn't
target-specific, then the :prop_dir:`ADDITIONAL_CLEAN_FILES` directory
property may be the more appropriate property to use.

Relative paths are allowed and are interpreted relative to the
current binary directory.

Contents of ``ADDITIONAL_CLEAN_FILES`` may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.

This property only works for the :generator:`Ninja` and the Makefile
generators.  It is ignored by other generators.
