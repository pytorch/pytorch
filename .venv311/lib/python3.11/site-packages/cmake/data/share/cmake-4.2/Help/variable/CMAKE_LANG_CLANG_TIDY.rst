CMAKE_<LANG>_CLANG_TIDY
-----------------------

.. versionadded:: 3.6

Default value for :prop_tgt:`<LANG>_CLANG_TIDY` target property
when ``<LANG>`` is ``C``, ``CXX``, ``OBJC`` or ``OBJCXX``.

This variable is used to initialize the property on each target as it is
created.  For example:

.. code-block:: cmake

  set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,readability-*)
  add_executable(foo foo.cxx)
