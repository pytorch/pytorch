XCODE_SCHEME_LAUNCH_MODE
------------------------

.. versionadded:: 3.25

Property value for ``Launch`` in the Info section of the generated Xcode
scheme.

Possible values are:

``AUTO``
  Launch automatically. This is the default.

``WAIT``
  Wait for the executable to be launched.

This property is initialized by the value of the variable
:variable:`CMAKE_XCODE_SCHEME_LAUNCH_MODE` if it is set when a target is
created.

Please refer to the :prop_tgt:`XCODE_GENERATE_SCHEME` target property
documentation to see all Xcode schema related properties.
