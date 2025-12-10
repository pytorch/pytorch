CUDA_ARCHITECTURES
------------------

.. versionadded:: 3.18

List of architectures to generate device code for.

An architecture can be suffixed by either ``-real`` or ``-virtual`` to specify
the kind of architecture to generate code for.
If no suffix is given then code is generated for both real and virtual
architectures.

A non-empty false value (e.g. ``OFF``) disables adding architectures.
This is intended to support packagers and rare cases where full control
over the passed flags is required.

This property is initialized by the value of the :variable:`CMAKE_CUDA_ARCHITECTURES`
variable if it is set when a target is created.

The ``CUDA_ARCHITECTURES`` target property must be set to a non-empty value on targets
that compile CUDA sources, or it is an error.  See policy :policy:`CMP0104`.

The ``CUDA_ARCHITECTURES`` may be set to one of the following special values:

``all``
  .. versionadded:: 3.23

  Compile for all supported major and minor real architectures,
  and the highest major virtual architecture.

``all-major``
  .. versionadded:: 3.23

  Compile for all supported major real architectures, and the highest
  major virtual architecture.

``native``
  .. versionadded:: 3.24

  Compile for the architecture(s) of the host's GPU(s).

Examples
^^^^^^^^

.. code-block:: cmake

  set_target_properties(tgt PROPERTIES CUDA_ARCHITECTURES "35;50;72")

Generates code for real and virtual architectures ``30``, ``50`` and ``72``.

.. code-block:: cmake

  set_property(TARGET tgt PROPERTY CUDA_ARCHITECTURES 70-real 72-virtual)

Generates code for real architecture ``70`` and virtual architecture ``72``.

.. code-block:: cmake

  set_property(TARGET tgt PROPERTY CUDA_ARCHITECTURES OFF)

CMake will not pass any architecture flags to the compiler.
