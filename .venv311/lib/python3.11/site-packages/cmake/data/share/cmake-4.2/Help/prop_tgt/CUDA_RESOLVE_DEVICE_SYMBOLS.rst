CUDA_RESOLVE_DEVICE_SYMBOLS
---------------------------

.. versionadded:: 3.9

CUDA only: Enables device linking for the specific library target where
required.

If set, this will tell the required compilers to enable device linking
on the library target. Device linking is an additional link step
required by some CUDA compilers when :prop_tgt:`CUDA_SEPARABLE_COMPILATION` is
enabled. Normally device linking is deferred until a shared library or
executable is generated, allowing for multiple static libraries to resolve
device symbols at the same time when they are used by a shared library or
executable.

If this property or :variable:`CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS` is unset,
static libraries are treated as if it is disabled while shared, module,
and executable targets behave as if it is on.

If :variable:`CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS` has been defined,
this property is initialized to the value the variable and overriding
the default behavior.

Note that device linking is not supported for :ref:`Object Libraries`.


For instance:

.. code-block:: cmake

  set_property(TARGET mystaticlib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
