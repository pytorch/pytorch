MACOSX_BUNDLE_INFO_PLIST
------------------------

Specify a custom ``Info.plist`` template for a macOS and iOS Application Bundle.

An executable target with :prop_tgt:`MACOSX_BUNDLE` enabled will be built as an
application bundle on macOS.  By default its ``Info.plist`` file is created
by configuring a template called ``MacOSXBundleInfo.plist.in`` located in the
:variable:`CMAKE_MODULE_PATH`.  This property specifies an alternative template
file name which may be a full path.

The following target properties may be set to specify content to be
configured into the file:

``MACOSX_BUNDLE_BUNDLE_NAME``
  Sets ``CFBundleName``.
``MACOSX_BUNDLE_BUNDLE_VERSION``
  Sets ``CFBundleVersion``.
``MACOSX_BUNDLE_COPYRIGHT``
  Sets ``NSHumanReadableCopyright``.
``MACOSX_BUNDLE_GUI_IDENTIFIER``
  Sets ``CFBundleIdentifier``.
``MACOSX_BUNDLE_ICON_FILE``
  Sets ``CFBundleIconFile``.
``MACOSX_BUNDLE_INFO_STRING``
  Sets ``CFBundleGetInfoString``.
``MACOSX_BUNDLE_LONG_VERSION_STRING``
  Sets ``CFBundleLongVersionString``.
``MACOSX_BUNDLE_SHORT_VERSION_STRING``
  Sets ``CFBundleShortVersionString``.

CMake variables of the same name may be set to affect all targets in a
directory that do not have each specific property set.  If a custom
``Info.plist`` is specified by this property it may of course hard-code
all the settings instead of using the target properties.
