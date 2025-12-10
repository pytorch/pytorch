CMAKE_CUDA_ARCHITECTURES
------------------------

.. versionadded:: 3.18

Default value for :prop_tgt:`CUDA_ARCHITECTURES` property of targets.

Initialized by the :envvar:`CUDAARCHS` environment variable if set.
Otherwise as follows depending on :variable:`CMAKE_CUDA_COMPILER_ID <CMAKE_<LANG>_COMPILER_ID>`:

- For ``Clang``: the oldest architecture that works.

- For ``NVIDIA``: the default architecture chosen by the compiler.
  See policy :policy:`CMP0104`.

Users are encouraged to override this, as the default varies across compilers
and compiler versions.

This variable is used to initialize the :prop_tgt:`CUDA_ARCHITECTURES` property
on all targets. See the target property for additional information.

Examples
^^^^^^^^

.. code-block:: cmake

  cmake_minimum_required(VERSION)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()

  project(example LANGUAGES CUDA)

``CMAKE_CUDA_ARCHITECTURES`` will default to ``75`` unless overridden by the user.
