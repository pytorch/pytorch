CMAKE_ANDROID_NDK_DEPRECATED_HEADERS
------------------------------------

.. versionadded:: 3.9

When :ref:`Cross Compiling for Android with the NDK`, this variable
may be set to specify whether to use the deprecated per-api-level
headers instead of the unified headers.

If not specified, the default will be *false* if using a NDK version
that provides the unified headers and *true* otherwise.
