CMAKE_FIND_LIBRARY_SUFFIXES
---------------------------

Suffixes to append when looking for libraries.

This specifies what suffixes to add to library names when the
:command:`find_library` command looks for libraries.  On Windows systems this
is typically ``.lib`` and, depending on the compiler, ``.dll.lib``, ``.dll.a``,
``.a`` (e.g. rustc, GCC, or Clang), so when it tries to find the ``foo``
library, it will look for ``[<prefix>]foo[.dll].lib`` and/or
``[<prefix>]foo[.dll].a``, depending on the compiler used and the ``<prefix>``
specified in the :variable:`CMAKE_FIND_LIBRARY_PREFIXES`.

.. include:: include/CMAKE_FIND_LIBRARY_VAR.rst
