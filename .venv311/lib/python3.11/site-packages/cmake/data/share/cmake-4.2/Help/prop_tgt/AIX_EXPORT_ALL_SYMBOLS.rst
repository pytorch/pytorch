AIX_EXPORT_ALL_SYMBOLS
----------------------

.. versionadded:: 3.17

On AIX, CMake automatically exports all symbols from shared libraries, and
from executables with the :prop_tgt:`ENABLE_EXPORTS` target property set.
Explicitly disable this boolean property to suppress the behavior and
export no symbols by default.  In this case it is expected that the project
will use other means to export some symbols.

This property is initialized by the value of
the :variable:`CMAKE_AIX_EXPORT_ALL_SYMBOLS` variable if it is set
when a target is created.
