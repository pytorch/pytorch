CMAKE_HIP_ARCHITECTURES
-----------------------

.. versionadded:: 3.21

List of GPU architectures to for which to generate device code.
Architecture names are interpreted based on :variable:`CMAKE_HIP_PLATFORM`.

This is initialized based on the value of :variable:`CMAKE_HIP_PLATFORM`:

``amd``
  Uses architectures reported by ``rocm_agent_enumerator``, if available,
  and otherwise to a default chosen by the compiler.

This variable is used to initialize the :prop_tgt:`HIP_ARCHITECTURES` property
on all targets. See the target property for additional information.
