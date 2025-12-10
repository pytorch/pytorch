CMAKE_ANDROID_ARCH
------------------

.. versionadded:: 3.4

When :ref:`Cross Compiling for Android with NVIDIA Nsight Tegra Visual Studio
Edition`, this variable may be set to specify the default value for the
:prop_tgt:`ANDROID_ARCH` target property.  See that target property for
additional information.

Otherwise, when :ref:`Cross Compiling for Android`, this variable provides
the name of the Android architecture corresponding to the value of the
:variable:`CMAKE_ANDROID_ARCH_ABI` variable.  The architecture name
may be one of:

* ``arm``
* ``arm64``
* ``mips``
* ``mips64``
* ``x86``
* ``x86_64``
