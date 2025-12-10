COMPILE_WARNING_AS_ERROR
------------------------

.. versionadded:: 3.24

Specify whether to treat warnings on compile as errors.
If enabled, adds a flag to treat warnings on compile as errors.
If the :option:`cmake --compile-no-warning-as-error` option is given
on the :manual:`cmake(1)` command line, this property is ignored.

This property is not implemented for all compilers.  It is silently ignored
if there is no implementation for the compiler being used.  The currently
implemented :variable:`compiler IDs <CMAKE_<LANG>_COMPILER_ID>` are:

* ``GNU``
* ``Clang``
* ``AppleClang``
* ``Fujitsu``
* ``FujitsuClang``
* ``IBMClang``
* ``Intel``
* ``IntelLLVM``
* ``LCC``
* ``MSVC``
* ``NVHPC``
* ``NVIDIA`` (CUDA)
* ``QCC``
* ``SunPro``
* ``Tasking``
* ``TI``
* ``VisualAge``
* ``XL``
* ``XLClang``

This property is initialized by the value of the variable
:variable:`CMAKE_COMPILE_WARNING_AS_ERROR` if it is set when a target is
created.
