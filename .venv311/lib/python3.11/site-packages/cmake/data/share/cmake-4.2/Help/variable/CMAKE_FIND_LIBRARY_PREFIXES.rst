CMAKE_FIND_LIBRARY_PREFIXES
---------------------------

Prefixes to prepend when looking for libraries.

This specifies what prefixes to add to library names when the
:command:`find_library` command looks for libraries.  On UNIX systems this is
typically ``lib``, meaning that when trying to find the ``foo`` library it
will look for ``libfoo``.

.. include:: include/CMAKE_FIND_LIBRARY_VAR.rst
