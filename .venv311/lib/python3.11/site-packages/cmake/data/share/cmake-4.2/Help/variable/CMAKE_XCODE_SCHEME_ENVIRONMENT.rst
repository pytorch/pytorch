CMAKE_XCODE_SCHEME_ENVIRONMENT
------------------------------

.. versionadded:: 3.17

Specify environment variables that should be added to the Arguments
section of the generated Xcode scheme.

If set to a list of environment variables and values of the form
``MYVAR=value`` those environment variables will be added to the
scheme.

This variable initializes the :prop_tgt:`XCODE_SCHEME_ENVIRONMENT`
property on all targets.

Please refer to the :prop_tgt:`XCODE_GENERATE_SCHEME` target property
documentation to see all Xcode schema related properties.
