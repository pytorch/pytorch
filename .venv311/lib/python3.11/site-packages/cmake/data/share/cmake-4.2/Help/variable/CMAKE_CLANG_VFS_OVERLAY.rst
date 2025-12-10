CMAKE_CLANG_VFS_OVERLAY
-----------------------

.. versionadded:: 3.19

When cross compiling for windows with clang-cl, this variable can be an
absolute path pointing to a clang virtual file system yaml file, which
will enable clang-cl to resolve windows header names on a case sensitive
file system.
