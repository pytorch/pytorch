HEADER_DIRS
-----------

.. versionadded:: 3.23

Semicolon-separated list of base directories of the target's default
header set (i.e. the file set with name and type ``HEADERS``). The property
supports :manual:`generator expressions <cmake-generator-expressions(7)>`.

This property is normally only set by :command:`target_sources(FILE_SET)`
rather than being manipulated directly.

See :prop_tgt:`HEADER_DIRS_<NAME>` for the list of base directories in
other header sets.
