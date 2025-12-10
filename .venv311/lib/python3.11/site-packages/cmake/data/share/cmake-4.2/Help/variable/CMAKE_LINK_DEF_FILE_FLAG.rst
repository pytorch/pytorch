CMAKE_LINK_DEF_FILE_FLAG
------------------------

Linker flag to be used to specify a ``.def`` file for dll creation.

The flag will be used to add a ``.def`` file when creating a dll on
Windows; this is only defined on Windows.

CMake sets this variable automatically during toolchain inspection by
calls to the :command:`project` or :command:`enable_language` commands.

If the per-language :variable:`CMAKE_<LANG>_LINK_DEF_FILE_FLAG` variable
is defined, it takes precedence over :variable:`!CMAKE_LINK_DEF_FILE_FLAG`.
