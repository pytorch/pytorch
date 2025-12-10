CMAKE_SCRIPT_MODE_FILE
----------------------

Full path to the :option:`cmake -P` script file currently being
processed.

When run in :option:`cmake -P` script mode, CMake sets this variable to
the full path of the script file.  When run to configure a ``CMakeLists.txt``
file, this variable is not set.

See Also
^^^^^^^^

* The :prop_gbl:`CMAKE_ROLE` global property provides the current running mode.
