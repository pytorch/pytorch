XCODE_EMIT_EFFECTIVE_PLATFORM_NAME
----------------------------------

.. versionadded:: 3.8

Control emission of ``EFFECTIVE_PLATFORM_NAME`` by the :generator:`Xcode`
generator.

It is required for building the same target with multiple SDKs. A
common use case is the parallel use of ``iphoneos`` and
``iphonesimulator`` SDKs.

Three different states possible that control when the :generator:`Xcode`
generator emits the ``EFFECTIVE_PLATFORM_NAME`` variable:

- If set to ``ON`` it will always be emitted
- If set to ``OFF`` it will never be emitted
- If unset (the default) it will only be emitted when the project was
  configured for an embedded Xcode SDK like iOS, tvOS, visionOS, watchOS
  or any of the simulators.

.. note::

  When this behavior is enable for generated Xcode projects, the
  ``EFFECTIVE_PLATFORM_NAME`` variable will leak into
  :manual:`Generator expressions <cmake-generator-expressions(7)>`
  like ``TARGET_FILE`` and will render those mostly unusable.
