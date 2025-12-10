CMAKE_FIND_PACKAGE_RESOLVE_SYMLINKS
-----------------------------------

.. versionadded:: 3.14

Set to ``TRUE`` to tell :command:`find_package` calls to resolve symbolic
links in the value of ``<PackageName>_DIR``.

This is helpful in use cases where the package search path points at a
proxy directory in which symlinks to the real package locations appear.
This is not enabled by default because there are also common use cases
in which the symlinks should be preserved.
