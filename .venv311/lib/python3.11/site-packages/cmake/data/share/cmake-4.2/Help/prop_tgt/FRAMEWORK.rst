FRAMEWORK
---------

Build ``SHARED`` or ``STATIC`` library as Framework Bundle on the macOS and iOS.

If such a library target has this property set to ``TRUE`` it will be
built as a framework when built on the macOS and iOS.  It will have the
directory structure required for a framework and will be suitable to
be used with the ``-framework`` option.  This property is initialized by the
value of the :variable:`CMAKE_FRAMEWORK` variable if it is set when a target is
created.

To customize ``Info.plist`` file in the framework, use
:prop_tgt:`MACOSX_FRAMEWORK_INFO_PLIST` target property.

For macOS see also the :prop_tgt:`FRAMEWORK_VERSION` target property.

Example of creation ``dynamicFramework``:

.. code-block:: cmake

  add_library(dynamicFramework SHARED
              dynamicFramework.c
              dynamicFramework.h
  )
  set_target_properties(dynamicFramework PROPERTIES
    FRAMEWORK TRUE
    FRAMEWORK_VERSION C
    MACOSX_FRAMEWORK_IDENTIFIER com.cmake.dynamicFramework
    MACOSX_FRAMEWORK_INFO_PLIST Info.plist
    # "current version" in semantic format in Mach-O binary file
    VERSION 16.4.0
    # "compatibility version" in semantic format in Mach-O binary file
    SOVERSION 1.0.0
    PUBLIC_HEADER dynamicFramework.h
    XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "iPhone Developer"
  )
