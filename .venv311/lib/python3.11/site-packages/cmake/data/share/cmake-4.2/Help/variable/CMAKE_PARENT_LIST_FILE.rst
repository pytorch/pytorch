CMAKE_PARENT_LIST_FILE
----------------------

Full path to the CMake file that included the current one.

While processing a CMake file loaded by :command:`include` or
:command:`find_package` this variable contains the full path to the file
including it.

While processing a ``CMakeLists.txt`` file, even in subdirectories,
this variable is not defined.  See policy :policy:`CMP0198`.

While processing a :option:`cmake -P` script, this variable is not defined
in the outermost script.

See also :variable:`CMAKE_CURRENT_LIST_FILE`.
