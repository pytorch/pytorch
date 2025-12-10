CMAKE_<LANG>_LINK_MODE
----------------------

.. versionadded:: 4.0

Defines how the link step is done. The possible values are:

``DRIVER``
  The compiler is used as driver for the link step.

``LINKER``
  The linker is used directly for the link step.

This variable is read-only. Setting it is undefined behavior.

See Also
^^^^^^^^

* The :variable:`CMAKE_<LANG>_USING_LINKER_<TYPE>` variable.
