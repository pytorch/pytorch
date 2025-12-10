HEADER_SET
----------

.. versionadded:: 3.23

Semicolon-separated list of files in the target's default header set,
(i.e. the file set with name and type ``HEADERS``). If any of the paths
are relative, they are computed relative to the target's source directory.
The property supports
:manual:`generator expressions <cmake-generator-expressions(7)>`.

This property is normally only set by :command:`target_sources(FILE_SET)`
rather than being manipulated directly.

See :prop_tgt:`HEADER_SET_<NAME>` for the list of files in other header sets.
