CMAKE_<LANG>_FLAGS
------------------

Language-wide flags for language ``<LANG>`` used when building for
all configurations.  These flags will be passed to all invocations
of the compiler.  This includes invocations that drive compiling
and those that drive linking.

For each language, if this variable is not defined, it is initialized
and stored in the cache using values from environment variables in
combination with CMake's builtin defaults for the toolchain:

* ``CMAKE_C_FLAGS``:
  Initialized by the :envvar:`CFLAGS` environment variable.
* ``CMAKE_CXX_FLAGS``:
  Initialized by the :envvar:`CXXFLAGS` environment variable.
* ``CMAKE_CUDA_FLAGS``:
  Initialized by the :envvar:`CUDAFLAGS` environment variable.
* ``CMAKE_Fortran_FLAGS``:
  Initialized by the :envvar:`FFLAGS` environment variable.
* ``CMAKE_CSharp_FLAGS``:
  Initialized by the :envvar:`CSFLAGS` environment variable.
* ``CMAKE_HIP_FLAGS``:
  Initialized by the :envvar:`HIPFLAGS` environment variable.
* ``CMAKE_ISPC_FLAGS``:
  Initialized by the :envvar:`ISPCFLAGS` environment variable.
* ``CMAKE_OBJC_FLAGS``:
  Initialized by the :envvar:`OBJCFLAGS` environment variable.
* ``CMAKE_OBJCXX_FLAGS``:
  Initialized by the :envvar:`OBJCXXFLAGS` environment variable.

This value is a command-line string fragment. Therefore, multiple options
should be separated by spaces, and options with spaces should be quoted.

The flags in this variable will be passed before those in the
per-configuration :variable:`CMAKE_<LANG>_FLAGS_<CONFIG>` variable.
On invocations driving compiling, flags from both variables will be passed
before flags added by commands such as :command:`add_compile_options` and
:command:`target_compile_options`. On invocations driving linking,
they will be passed before flags added by commands such as
:command:`add_link_options` and :command:`target_link_options`.
