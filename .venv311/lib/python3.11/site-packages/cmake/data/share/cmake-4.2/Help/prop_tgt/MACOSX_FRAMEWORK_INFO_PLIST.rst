MACOSX_FRAMEWORK_INFO_PLIST
---------------------------

Specify a custom ``Info.plist`` template for a macOS and iOS Framework.

A library target with :prop_tgt:`FRAMEWORK` enabled will be built as a
framework on macOS.  By default its ``Info.plist`` file is created by
configuring a template called ``MacOSXFrameworkInfo.plist.in`` located in the
:variable:`CMAKE_MODULE_PATH`.  This property specifies an alternative template
file name which may be a full path.

The following target properties may be set to specify content to be
configured into the file:

``MACOSX_FRAMEWORK_BUNDLE_NAME``
  .. versionadded:: 3.31

  Sets ``CFBundleName``.

``MACOSX_FRAMEWORK_BUNDLE_VERSION``
  Sets ``CFBundleVersion``.

``MACOSX_FRAMEWORK_ICON_FILE``
  Sets ``CFBundleIconFile``.

``MACOSX_FRAMEWORK_IDENTIFIER``
  Sets ``CFBundleIdentifier``.

``MACOSX_FRAMEWORK_SHORT_VERSION_STRING``
  Sets ``CFBundleShortVersionString``.

CMake variables of the same name may be set to affect all targets in a
directory that do not have each specific property set.  If a custom
``Info.plist`` is specified by this property it may of course hard-code
all the settings instead of using the target properties.
