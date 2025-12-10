ANDROID_API
-----------

.. versionadded:: 3.1

When :ref:`Cross Compiling for Android with NVIDIA Nsight Tegra Visual Studio
Edition`, this property sets the Android target API version (e.g. ``15``).
The version number must be a positive decimal integer.  This property is
initialized by the value of the :variable:`CMAKE_ANDROID_API` variable if
it is set when a target is created.
