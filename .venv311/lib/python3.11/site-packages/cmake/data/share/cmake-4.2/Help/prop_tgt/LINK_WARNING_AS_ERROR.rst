LINK_WARNING_AS_ERROR
---------------------

.. versionadded:: 4.0

Specify whether to treat warnings on link as errors.
If enabled, adds a flag to treat warnings on link as errors.
If the :option:`cmake --link-no-warning-as-error` option is given
on the :manual:`cmake(1)` command line, this property is ignored.

This property takes a :ref:`semicolon-separated-list <CMake Language Lists>` of
the following values:

* ``LINKER``: treat the linker warnings as errors.
* ``DRIVER``: treat the compiler warnings as errors when used to drive the link
  step. See the :prop_tgt:`COMPILE_WARNING_AS_ERROR` target property for more
  information.

Moreover, for consistency with the :prop_tgt:`COMPILE_WARNING_AS_ERROR` target
property, a boolean value can be specified:

* ``True`` value: this is equivalent to ``LINKER`` and ``DRIVER`` values.
* ``False`` value: deactivate this feature for the target.

This property is not implemented for all linkers.  It is silently ignored
if there is no implementation for the linker being used.  The currently
implemented :variable:`compiler linker IDs <CMAKE_<LANG>_COMPILER_LINKER_ID>`
are:

* ``AIX``
* ``AppleClang``
* ``GNU``
* ``GNUgold``
* ``LLD``
* ``MOLD``
* ``MSVC``
* ``Solaris``

This property is initialized by the value of the variable
:variable:`CMAKE_LINK_WARNING_AS_ERROR` if it is set when a target is
created.
