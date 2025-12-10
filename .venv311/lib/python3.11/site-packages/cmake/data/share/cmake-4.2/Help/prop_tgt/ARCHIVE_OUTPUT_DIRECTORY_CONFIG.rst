ARCHIVE_OUTPUT_DIRECTORY_<CONFIG>
---------------------------------

Per-configuration output directory for
:ref:`ARCHIVE <Archive Output Artifacts>` target files.

This is a per-configuration version of the
:prop_tgt:`ARCHIVE_OUTPUT_DIRECTORY` target property, but
multi-configuration generators (:ref:`Visual Studio <Visual Studio
Generators>`, :generator:`Xcode`, :generator:`Ninja Multi-Config`) do NOT
append a per-configuration subdirectory to the specified directory.  This
property is initialized by the value of the
:variable:`CMAKE_ARCHIVE_OUTPUT_DIRECTORY_<CONFIG>` variable if
it is set when a target is created.

Contents of ``ARCHIVE_OUTPUT_DIRECTORY_<CONFIG>`` may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.
