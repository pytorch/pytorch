CMAKE_OSX_DEPLOYMENT_TARGET
---------------------------

Specify the minimum version of the target platform, e.g., macOS or iOS,
on which the target binaries are to be deployed.

For builds targeting macOS (:variable:`CMAKE_SYSTEM_NAME` is ``Darwin``), if
``CMAKE_OSX_DEPLOYMENT_TARGET`` is not explicitly set, a default is set:

* If the ``MACOSX_DEPLOYMENT_TARGET`` environment variable is non-empty,
  its value is the default.

* Otherwise, if using the :generator:`Xcode` generator, and the host's
  macOS version is older than the macOS SDK (:variable:`CMAKE_OSX_SYSROOT`,
  if set, or Xcode's default SDK), the host's macOS version is the default.

  .. versionchanged:: 4.0

    Previously this was done for all generators, not just Xcode.

* Otherwise, the default is empty.

The effects of ``CMAKE_OSX_DEPLOYMENT_TARGET`` depend on the generator:

:generator:`Xcode`

  If ``CMAKE_OSX_DEPLOYMENT_TARGET`` is set to a non-empty value, it is added
  to the generated Xcode project as the ``MACOSX_DEPLOYMENT_TARGET`` setting.
  Otherwise, no such setting is added, so Xcode's default deployed target is
  used, typically based on the SDK version.

Other Generators

  If ``CMAKE_OSX_DEPLOYMENT_TARGET`` is set to a non-empty value, it is passed
  to the compiler via the ``-mmacosx-version-min`` flag or equivalent.
  Otherwise, no such flag is added, so the compiler's default deployment
  target is used.

.. include:: include/CMAKE_OSX_VARIABLE.rst
