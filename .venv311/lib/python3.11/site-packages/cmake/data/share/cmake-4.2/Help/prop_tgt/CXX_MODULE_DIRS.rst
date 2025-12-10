CXX_MODULE_DIRS
---------------

.. versionadded:: 3.28

Semicolon-separated list of base directories of the target's default
C++ module set (i.e. the file set with name and type ``CXX_MODULES``). The
property supports
:manual:`generator expressions <cmake-generator-expressions(7)>`.

This property is normally only set by :command:`target_sources(FILE_SET)`
rather than being manipulated directly.

See :prop_tgt:`CXX_MODULE_DIRS_<NAME>` for the list of base directories in
other C++ module sets.
