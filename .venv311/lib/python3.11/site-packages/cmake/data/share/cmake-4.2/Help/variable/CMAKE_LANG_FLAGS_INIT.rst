CMAKE_<LANG>_FLAGS_INIT
-----------------------

.. versionadded:: 3.7

Value used to initialize the :variable:`CMAKE_<LANG>_FLAGS` cache entry
the first time a build tree is configured for language ``<LANG>``.
This variable is meant to be set by a :variable:`toolchain file
<CMAKE_TOOLCHAIN_FILE>`.  CMake may prepend or append content to
the value based on the environment and target platform.  For example,
the contents of a ``xxxFLAGS`` environment variable will be prepended,
where ``xxx`` will be language-specific but not necessarily the same as
``<LANG>`` (e.g. :envvar:`CXXFLAGS` for ``CXX``, :envvar:`FFLAGS` for
``Fortran``, and so on).
This value is a command-line string fragment. Therefore, multiple options
should be separated by spaces, and options with spaces should be quoted.

See also the configuration-specific
:variable:`CMAKE_<LANG>_FLAGS_<CONFIG>_INIT` variable.
