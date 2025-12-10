CMAKE_HIP_PLATFORM
------------------

.. versionadded:: 3.28

GPU platform for which HIP language sources are to be compiled.

The value must be one of:

``amd``
  AMD GPUs

``nvidia``
  NVIDIA GPUs

If not specified, a default is computed via ``hipconfig --platform``.

:variable:`CMAKE_HIP_ARCHITECTURES` entries are interpreted with
as architectures of the GPU platform.

:variable:`CMAKE_HIP_COMPILER <CMAKE_<LANG>_COMPILER>` must target
the same GPU platform.
