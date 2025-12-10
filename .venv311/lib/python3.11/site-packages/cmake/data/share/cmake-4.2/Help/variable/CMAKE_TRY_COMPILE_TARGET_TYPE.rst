CMAKE_TRY_COMPILE_TARGET_TYPE
-----------------------------

.. versionadded:: 3.6

Type of target generated for :command:`try_compile` calls using the
source file signature.  Valid values are:

``EXECUTABLE``
  Use :command:`add_executable` to name the source file in the
  generated project.  This is the default if no value is given.

``STATIC_LIBRARY``
  Use :command:`add_library` with the ``STATIC`` option to name the
  source file in the generated project.  This avoids running the
  linker and is intended for use with cross-compiling toolchains
  that cannot link without custom flags or linker scripts.
