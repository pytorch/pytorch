CMP0081
-------

.. versionadded:: 3.13

Relative paths not allowed in :prop_tgt:`LINK_DIRECTORIES` target property.

CMake 3.12 and lower allowed the :prop_dir:`LINK_DIRECTORIES` directory
property to contain relative paths.  The base path for such relative
entries is not well defined.  CMake 3.13 and later will issue a
``FATAL_ERROR`` if the :prop_tgt:`LINK_DIRECTORIES` target property
(which is initialized by the :prop_dir:`LINK_DIRECTORIES` directory property)
contains a relative path.

The ``OLD`` behavior for this policy is not to warn about relative paths
in the :prop_tgt:`LINK_DIRECTORIES` target property.  The ``NEW`` behavior for
this policy is to issue a ``FATAL_ERROR`` if :prop_tgt:`LINK_DIRECTORIES`
contains a relative path.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 3.13
.. |WARNS_OR_DOES_NOT_WARN| replace:: warns
.. include:: include/STANDARD_ADVICE.rst

.. include:: include/DEPRECATED.rst
