CMAKE_<LANG>_DEVICE_LINK_MODE
-----------------------------

.. versionadded:: 4.0

Defines how the device link step is done. The possible values are:

``DRIVER``
  The compiler is used as driver for the device link step.

``LINKER``
  The linker is used directly for the device link step.

This variable is read-only. Setting it is undefined behavior.
