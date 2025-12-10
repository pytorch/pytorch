HIP_ARCHITECTURES
-----------------

.. versionadded:: 3.21

List of GPU architectures to for which to generate device code.
Architecture names are interpreted based on :variable:`CMAKE_HIP_PLATFORM`.

A non-empty false value (e.g. ``OFF``) disables adding architectures.
This is intended to support packagers and rare cases where full control
over the passed flags is required.

This property is initialized by the value of the :variable:`CMAKE_HIP_ARCHITECTURES`
variable if it is set when a target is created.

The HIP compilation model has two modes: whole and separable. Whole compilation
generates device code at compile time. Separable compilation generates device
code at link time. Therefore the ``HIP_ARCHITECTURES`` target property should
be set on targets that compile or link with any HIP sources.

Examples
^^^^^^^^

.. code-block:: cmake

  set_property(TARGET tgt PROPERTY HIP_ARCHITECTURES gfx801 gfx900)

Generates code for both ``gfx801`` and ``gfx900``.
