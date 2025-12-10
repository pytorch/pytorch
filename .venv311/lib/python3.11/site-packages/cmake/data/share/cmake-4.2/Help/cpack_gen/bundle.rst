CPack Bundle Generator
----------------------

CPack Bundle generator (macOS) specific options

Variables specific to CPack Bundle generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Installers built on macOS using the Bundle generator use the
aforementioned DragNDrop (``CPACK_DMG_xxx``) variables, plus the following
Bundle-specific parameters (``CPACK_BUNDLE_xxx``).

.. variable:: CPACK_BUNDLE_NAME

 The name of the generated bundle. This appears in the macOS Finder as the
 bundle name. Required.

.. variable:: CPACK_BUNDLE_PLIST

 Path to an macOS Property List (``.plist``) file that will be used
 for the generated bundle. This
 assumes that the caller has generated or specified their own ``Info.plist``
 file. Required.

.. variable:: CPACK_BUNDLE_ICON

 Path to an macOS icon file that will be used as the icon for the generated
 bundle. This is the icon that appears in the macOS Finder for the bundle, and
 in the macOS dock when the bundle is opened. Required.

.. variable:: CPACK_BUNDLE_STARTUP_COMMAND

 Path to a startup script. This is a path to an executable or script that
 will be run whenever an end-user double-clicks the generated bundle in the
 macOS Finder. Optional.

.. variable:: CPACK_BUNDLE_APPLE_CERT_APP

 .. versionadded:: 3.2

 The name of your Apple supplied code signing certificate for the application.
 The name usually takes the form ``Developer ID Application: [Name]`` or
 ``3rd Party Mac Developer Application: [Name]``. If this variable is not set
 the application will not be signed.

.. variable:: CPACK_BUNDLE_APPLE_ENTITLEMENTS

 .. versionadded:: 3.2

 The name of the Property List (``.plist``) file that contains your Apple
 entitlements for sandboxing your application. This file is required
 for submission to the macOS App Store.

.. variable:: CPACK_BUNDLE_APPLE_CODESIGN_FILES

 .. versionadded:: 3.2

 A list of additional files that you wish to be signed. You do not need to
 list the main application folder, or the main executable. You should
 list any frameworks and plugins that are included in your app bundle.

.. variable:: CPACK_BUNDLE_APPLE_CODESIGN_PARAMETER

 .. versionadded:: 3.3

 Additional parameter that will passed to ``codesign``.
 Default value: ``--deep -f``

.. variable:: CPACK_COMMAND_CODESIGN

 .. versionadded:: 3.2

 Path to the ``codesign(1)`` command used to sign applications with an
 Apple cert. This variable can be used to override the automatically
 detected command (or specify its location if the auto-detection fails
 to find it).
