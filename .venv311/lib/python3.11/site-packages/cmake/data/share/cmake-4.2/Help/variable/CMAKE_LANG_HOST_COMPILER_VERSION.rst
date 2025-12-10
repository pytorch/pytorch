CMAKE_<LANG>_HOST_COMPILER_VERSION
----------------------------------

.. versionadded:: 3.31

This variable is available when ``<LANG>`` is ``CUDA`` or ``HIP``
and :variable:`CMAKE_<LANG>_COMPILER_ID` is ``NVIDIA``.
It contains the version of the host compiler invoked by ``nvcc``,
either by default or as specified by :variable:`CMAKE_<LANG>_HOST_COMPILER`,
in the same format as :variable:`CMAKE_<LANG>_COMPILER_VERSION`.
