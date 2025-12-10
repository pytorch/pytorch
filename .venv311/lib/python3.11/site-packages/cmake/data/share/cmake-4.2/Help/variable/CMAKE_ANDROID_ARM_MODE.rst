CMAKE_ANDROID_ARM_MODE
----------------------

.. versionadded:: 3.7

When :ref:`Cross Compiling for Android` and :variable:`CMAKE_ANDROID_ARCH_ABI`
is set to one of the ``armeabi`` architectures, set ``CMAKE_ANDROID_ARM_MODE``
to ``ON`` to target 32-bit ARM processors (``-marm``).  Otherwise, the
default is to target the 16-bit Thumb processors (``-mthumb``).
