CMAKE_<LANG>_ARCHIVER_WRAPPER_FLAG_SEP
--------------------------------------

.. versionadded:: 4.0

This variable is used with :variable:`CMAKE_<LANG>_ARCHIVER_WRAPPER_FLAG`
variable to format ``ARCHIVER:`` prefix in the static library options
(see :prop_tgt:`STATIC_LIBRARY_OPTIONS`).

When specified, arguments of the ``ARCHIVER:`` prefix will be concatenated
using this value as separator.
