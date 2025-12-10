CMAKE_ANDROID_API
-----------------

.. versionadded:: 3.1

When :ref:`Cross Compiling for Android with NVIDIA Nsight Tegra Visual Studio
Edition`, this variable may be set to specify the default value for the
:prop_tgt:`ANDROID_API` target property.  See that target property for
additional information.

When :ref:`Cross Compiling for Android`, the :variable:`CMAKE_SYSTEM_VERSION`
variable represents the Android API version number targeted.  For historical
reasons, if a toolchain file sets ``CMAKE_ANDROID_API``, but not
``CMAKE_SYSTEM_VERSION``, the latter will be initialized using the former.
