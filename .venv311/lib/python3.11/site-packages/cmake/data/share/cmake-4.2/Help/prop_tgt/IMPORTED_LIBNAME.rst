IMPORTED_LIBNAME
----------------

.. versionadded:: 3.8

Specify the link library name for an :ref:`imported <Imported Targets>`
:ref:`Interface Library <Interface Libraries>`.

An interface library builds no library file itself but does specify
usage requirements for its consumers.  The ``IMPORTED_LIBNAME``
property may be set to specify a single library name to be placed
on the link line in place of the interface library target name as
a requirement for using the interface.

This property is intended for use in naming libraries provided by
a platform SDK for which the full path to a library file may not
be known.  The value may be a plain library name such as ``foo``
but may *not* be a path (e.g. ``/usr/lib/libfoo.so``) or a flag
(e.g. ``-Wl,...``).  The name is never treated as a library target
name even if it happens to name one.

The ``IMPORTED_LIBNAME`` property is allowed only on
:ref:`imported <Imported Targets>` :ref:`Interface Libraries`
and is rejected on targets of other types (for which
the :prop_tgt:`IMPORTED_LOCATION` target property may be used).
