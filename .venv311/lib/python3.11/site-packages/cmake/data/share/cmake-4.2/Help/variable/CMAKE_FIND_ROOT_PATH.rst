CMAKE_FIND_ROOT_PATH
--------------------

:ref:`Semicolon-separated list <CMake Language Lists>` of root paths to search on the filesystem.

This variable is most useful when cross-compiling. CMake uses the paths in
this list as alternative roots to find filesystem items with
:command:`find_package`, :command:`find_library` etc.
