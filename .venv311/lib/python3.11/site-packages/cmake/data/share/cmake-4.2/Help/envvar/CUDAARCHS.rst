CUDAARCHS
---------

.. versionadded:: 3.20

.. include:: include/ENV_VAR.rst

Value used to initialize :variable:`CMAKE_CUDA_ARCHITECTURES` on the first
configuration. Subsequent runs will use the value stored in the cache.

This is a semicolon-separated list of architectures as described in
:prop_tgt:`CUDA_ARCHITECTURES`.
