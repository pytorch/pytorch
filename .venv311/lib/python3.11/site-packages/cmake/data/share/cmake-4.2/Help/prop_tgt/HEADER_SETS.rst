HEADER_SETS
-----------

.. versionadded:: 3.23

Read-only list of the target's ``PRIVATE`` and ``PUBLIC`` header sets (i.e.
all file sets with the type ``HEADERS``). Files listed in these file sets are
treated as source files for the purpose of IDE integration. The files also
have their :prop_sf:`HEADER_FILE_ONLY` property set to ``TRUE``.

Header sets may be defined using the :command:`target_sources` command
``FILE_SET`` option with type ``HEADERS``.

See also :prop_tgt:`HEADER_SET_<NAME>`, :prop_tgt:`HEADER_SET` and
:prop_tgt:`INTERFACE_HEADER_SETS`.
