CMAKE_<LANG>_LINKER_LAUNCHER
----------------------------

.. versionadded:: 3.21

Default value for :prop_tgt:`<LANG>_LINKER_LAUNCHER` target property. This
variable is used to initialize the property on each target as it is created.
This is done only when ``<LANG>`` is one of:

* ``C``

* ``CXX``

* ``CUDA``

  .. versionadded:: 4.1

* ``OBJC``

* ``OBJCXX``

* ``Fortran``

  .. versionadded:: 4.1

* ``HIP``

  .. versionadded:: 4.1

This variable is initialized to the :envvar:`CMAKE_<LANG>_LINKER_LAUNCHER`
environment variable if it is set.
