ANDROID_API_MIN
---------------

.. versionadded:: 3.2

Set the Android MIN API version (e.g. ``9``).  The version number
must be a positive decimal integer.  This property is initialized by
the value of the :variable:`CMAKE_ANDROID_API_MIN` variable if it is set
when a target is created.  Native code builds using this API version.
