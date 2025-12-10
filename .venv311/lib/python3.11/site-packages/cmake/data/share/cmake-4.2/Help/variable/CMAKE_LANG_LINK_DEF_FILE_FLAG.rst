CMAKE_<LANG>_LINK_DEF_FILE_FLAG
-------------------------------

.. versionadded:: 4.1

Linker flag to be used to specify a ``.def`` file for dll creation
with the toolchain for language ``<LANG>``.

CMake sets this variable automatically during toolchain inspection by
calls to the :command:`project` or :command:`enable_language` commands.

If the :variable:`!CMAKE_<LANG>_LINK_DEF_FILE_FLAG` variable
is defined, it takes precedence over the language-agnostic
:variable:`CMAKE_LINK_DEF_FILE_FLAG` variable.
