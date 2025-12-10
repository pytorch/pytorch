CMAKE_<LANG>_HOST_COMPILER
--------------------------

.. versionadded:: 3.10
  ``CMAKE_CUDA_HOST_COMPILER``

.. versionadded:: 3.28
  ``CMAKE_HIP_HOST_COMPILER``

This variable is available when ``<LANG>`` is ``CUDA`` or ``HIP``.

When :variable:`CMAKE_<LANG>_COMPILER_ID` is
``NVIDIA``, ``CMAKE_<LANG>_HOST_COMPILER`` selects the compiler executable
to use when compiling host code for ``CUDA`` or ``HIP`` language files.
This maps to the ``nvcc -ccbin`` option.

The ``CMAKE_<LANG>_HOST_COMPILER`` variable may be set explicitly before CUDA
or HIP is first enabled by a :command:`project` or :command:`enable_language`
command.
This can be done via ``-DCMAKE_<LANG>_HOST_COMPILER=...`` on the command line
or in a :ref:`toolchain file <Cross Compiling Toolchain>`.  Or, one may set
the :envvar:`CUDAHOSTCXX` or :envvar:`HIPHOSTCXX` environment variable to
provide a default value.

Once the CUDA or HIP language is enabled, the ``CMAKE_<LANG>_HOST_COMPILER``
variable is read-only and changes to it are undefined behavior.

.. note::

  Since ``CMAKE_<LANG>_HOST_COMPILER`` is meaningful only when the
  :variable:`CMAKE_<LANG>_COMPILER_ID` is ``NVIDIA``,
  it does not make sense to set ``CMAKE_<LANG>_HOST_COMPILER`` without also
  setting ``CMAKE_<LANG>_COMPILER`` to NVCC.

.. note::

  Projects should not try to set ``CMAKE_<LANG>_HOST_COMPILER`` to match
  :variable:`CMAKE_CXX_COMPILER <CMAKE_<LANG>_COMPILER>` themselves.
  It is the end-user's responsibility, not the project's, to ensure that
  NVCC targets the same ABI as the C++ compiler.

.. note::

  Ignored when using :ref:`Visual Studio Generators`.

See the :variable:`CMAKE_<LANG>_HOST_COMPILER_ID` and
:variable:`CMAKE_<LANG>_HOST_COMPILER_VERSION` variables for
information about the host compiler used by ``nvcc``, whether
by default or specified by ``CMAKE_<LANG>_HOST_COMPILER``.
