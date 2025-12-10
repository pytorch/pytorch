IOS_INSTALL_COMBINED
--------------------

.. versionadded:: 3.5
.. deprecated:: 3.28

  :prop_tgt:`IOS_INSTALL_COMBINED` was designed to make universal binaries
  containing iOS/arm* device code paired with iOS Simulator/x86_64 code
  (or similar for other Apple embedded platforms). Universal binaries can only
  differentiate code based on CPU type, so this only made sense before the
  days of arm64 macOS machines (i.e. iOS Simulator/arm64). Apple now
  recommends xcframeworks, which contain multiple binaries for different
  platforms, for this use case.

Build a combined (device and simulator) target when installing.

When this property is set to false, which is the default, then it will
either be built with the device SDK or the simulator SDK depending on the SDK
set. But if this property is set to true then the target will at install time
also be built for the other SDK and combined into one library.

.. note::

  If a selected architecture is available for both device SDK and simulator
  SDK it will be built for the SDK selected by :variable:`CMAKE_OSX_SYSROOT`
  and removed from the other SDK.

This feature requires at least Xcode version 6.
