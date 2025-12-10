CMAKE_<LANG>_ANDROID_TOOLCHAIN_PREFIX
-------------------------------------

.. versionadded:: 3.7

When :ref:`Cross Compiling for Android` this variable contains the absolute
path prefixing the toolchain GNU compiler and its binutils.

See also :variable:`CMAKE_<LANG>_ANDROID_TOOLCHAIN_SUFFIX`
and :variable:`CMAKE_<LANG>_ANDROID_TOOLCHAIN_MACHINE`.

For example, the path to the linker is:

.. code-block:: cmake

  ${CMAKE_CXX_ANDROID_TOOLCHAIN_PREFIX}ld${CMAKE_CXX_ANDROID_TOOLCHAIN_SUFFIX}
