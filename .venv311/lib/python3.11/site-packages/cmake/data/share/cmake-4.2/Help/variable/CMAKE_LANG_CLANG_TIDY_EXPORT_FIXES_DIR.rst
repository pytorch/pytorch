CMAKE_<LANG>_CLANG_TIDY_EXPORT_FIXES_DIR
----------------------------------------

.. versionadded:: 3.26

Default value for :prop_tgt:`<LANG>_CLANG_TIDY_EXPORT_FIXES_DIR` target
property when ``<LANG>`` is ``C``, ``CXX``, ``OBJC`` or ``OBJCXX``.

This variable is used to initialize the property on each target as it is
created.  For example:

.. code-block:: cmake

  set(CMAKE_CXX_CLANG_TIDY_EXPORT_FIXES_DIR clang-tidy-fixes)
  add_executable(foo foo.cxx)
