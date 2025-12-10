OBJC_STANDARD
-------------

.. versionadded:: 3.16

The OBJC standard whose features are requested to build this target.

This property specifies the OBJC standard whose features are requested
to build this target.  For some compilers, this results in adding a
flag such as ``-std=gnu11`` to the compile line.

Supported values are:

``90``
  Objective C89/C90

``99``
  Objective C99

``11``
  Objective C11

``17``
  .. versionadded:: 3.21

  Objective C17

``23``
  .. versionadded:: 3.21

  Objective C23

If the value requested does not result in a compile flag being added for
the compiler in use, a previous standard flag will be added instead.  This
means that using:

.. code-block:: cmake

  set_property(TARGET tgt PROPERTY OBJC_STANDARD 11)

with a compiler which does not support ``-std=gnu11`` or an equivalent
flag will not result in an error or warning, but will instead add the
``-std=gnu99`` or ``-std=gnu90`` flag if supported.  This "decay" behavior may
be controlled with the :prop_tgt:`OBJC_STANDARD_REQUIRED` target property.
Additionally, the :prop_tgt:`OBJC_EXTENSIONS` target property may be used to
control whether compiler-specific extensions are enabled on a per-target basis.

If the property is not set, and the project has set the :prop_tgt:`C_STANDARD`,
the value of :prop_tgt:`C_STANDARD` is set for ``OBJC_STANDARD``.

See the :manual:`cmake-compile-features(7)` manual for information on
compile features and a list of supported compilers.

This property is initialized by the value of
the :variable:`CMAKE_OBJC_STANDARD` variable if it is set when a target
is created.
