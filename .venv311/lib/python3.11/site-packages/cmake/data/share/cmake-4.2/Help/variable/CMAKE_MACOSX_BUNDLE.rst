CMAKE_MACOSX_BUNDLE
-------------------

Default value for :prop_tgt:`MACOSX_BUNDLE` of targets.

This variable is used to initialize the :prop_tgt:`MACOSX_BUNDLE` property on
all the targets.  See that target property for additional information.

This variable is set to ``ON`` by default if :variable:`CMAKE_SYSTEM_NAME`
equals to :ref:`iOS, tvOS, visionOS or watchOS <Cross Compiling for iOS, tvOS, visionOS, or watchOS>`.
