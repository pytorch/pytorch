CMP0073
-------

.. versionadded:: 3.12

Do not produce legacy ``_LIB_DEPENDS`` cache entries.

Ancient CMake versions once used ``<tgt>_LIB_DEPENDS`` cache entries to
propagate library link dependencies.  This has long been done by other
means, leaving the :command:`export_library_dependencies` command as the
only user of these values.  That command has long been disallowed by
policy :policy:`CMP0033`, but the ``<tgt>_LIB_DEPENDS`` cache entries
were left for compatibility with possible non-standard uses by projects.

CMake 3.12 and above now prefer to not produce these cache entries
at all.  This policy provides compatibility with projects that have
not been updated to avoid using them.

The ``OLD`` behavior for this policy is to set ``<tgt>_LIB_DEPENDS`` cache
entries.  The ``NEW`` behavior for this policy is to not set them.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 3.12
.. |WARNS_OR_DOES_NOT_WARN| replace:: does *not* warn
.. include:: include/STANDARD_ADVICE.rst

.. include:: include/DEPRECATED.rst
