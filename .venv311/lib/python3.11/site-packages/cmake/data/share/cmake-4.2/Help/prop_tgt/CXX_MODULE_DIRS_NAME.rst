CXX_MODULE_DIRS_<NAME>
----------------------

.. versionadded:: 3.28

Semicolon-separated list of base directories of the target's ``<NAME>`` C++
module set, which has the set type ``CXX_MODULES``. The property supports
:manual:`generator expressions <cmake-generator-expressions(7)>`.

This property is normally only set by :command:`target_sources(FILE_SET)`
rather than being manipulated directly.

See :prop_tgt:`CXX_MODULE_DIRS` for the list of base directories in the
default C++ module set. See :prop_tgt:`CXX_MODULE_SETS` for the file set names
of all C++ module sets.
