CMAKE_XCODE_SCHEME_ENABLE_GPU_FRAME_CAPTURE_MODE
------------------------------------------------

.. versionadded:: 3.23

Populate ``GPU Frame Capture`` in the Options section of
the generated Xcode scheme. Example values are ``Metal`` and
``Disabled``.

This variable initializes the
:prop_tgt:`XCODE_SCHEME_ENABLE_GPU_FRAME_CAPTURE_MODE`
property on all targets.

Please refer to the :prop_tgt:`XCODE_GENERATE_SCHEME` target property
documentation to see all Xcode schema related properties.
