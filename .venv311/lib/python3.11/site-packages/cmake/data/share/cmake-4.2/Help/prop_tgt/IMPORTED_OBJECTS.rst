IMPORTED_OBJECTS
----------------

.. versionadded:: 3.9

A :ref:`semicolon-separated list <CMake Language Lists>` of absolute paths
to the object files on disk for an :ref:`imported <Imported targets>`
:ref:`object library <object libraries>`.

Ignored for non-imported targets.

Projects may skip ``IMPORTED_OBJECTS`` if the configuration-specific
property :prop_tgt:`IMPORTED_OBJECTS_<CONFIG>` is set instead, except in
situations as noted in the section below.


Xcode Generator Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.20

For Apple platforms, a project may be built for more than one architecture.
This is controlled by the :variable:`CMAKE_OSX_ARCHITECTURES` variable.
For all but the :generator:`Xcode` generator, CMake invokes compilers once
per source file and passes multiple ``-arch`` flags, leading to a single
object file which will be a universal binary.  Such object files work well
when listed in the ``IMPORTED_OBJECTS`` of a separate CMake build, even for
the :generator:`Xcode` generator.  But producing such object files with the
:generator:`Xcode` generator is more difficult, since it invokes the compiler
once per architecture for each source file.  Unlike the other generators,
it does not generate universal object file binaries.

A further complication with the :generator:`Xcode` generator is that when
targeting device platforms (iOS, tvOS, visionOS or watchOS), the :generator:`Xcode`
generator has the ability to use either the device or simulator SDK without
needing CMake to be re-run.  The SDK can be selected at build time.
But since some architectures can be supported by both the device and the
simulator SDKs (e.g. ``arm64`` with Xcode 12 or later), not all combinations
can be represented in a single universal binary.  The only solution in this
case is to have multiple object files.

``IMPORTED_OBJECTS`` doesn't support generator expressions, so every file
it lists needs to be valid for every architecture and SDK.  If incorporating
object files that are not universal binaries, the path and/or file name of
each object file has to somehow encapsulate the different architectures and
SDKs.  With the :generator:`Xcode` generator, Xcode variables of the form
``$(...)`` can be used to represent these aspects and Xcode will substitute
the appropriate values at build time.  CMake doesn't interpret these
variables and embeds them unchanged in the Xcode project file.
``$(CURRENT_ARCH)`` can be used to represent the architecture, while
``$(EFFECTIVE_PLATFORM_NAME)`` can be used to differentiate between SDKs.

The following shows one example of how these two variables can be used to
refer to an object file whose location depends on both the SDK and the
architecture:

.. code-block:: cmake

  add_library(someObjs OBJECT IMPORTED)

  set_property(TARGET someObjs PROPERTY IMPORTED_OBJECTS
    # Quotes are required because of the ()
    "/path/to/somewhere/objects$(EFFECTIVE_PLATFORM_NAME)/$(CURRENT_ARCH)/func.o"
  )

  # Example paths:
  #   /path/to/somewhere/objects-iphoneos/arm64/func.o
  #   /path/to/somewhere/objects-iphonesimulator/x86_64/func.o

In some cases, you may want to have configuration-specific object files
as well.  The ``$(CONFIGURATION)`` Xcode variable is often used for this and
can be used in conjunction with the others mentioned above:

.. code-block:: cmake

  add_library(someObjs OBJECT IMPORTED)
  set_property(TARGET someObjs PROPERTY IMPORTED_OBJECTS
    "/path/to/somewhere/$(CONFIGURATION)$(EFFECTIVE_PLATFORM_NAME)/$(CURRENT_ARCH)/func.o"
  )

  # Example paths:
  #   /path/to/somewhere/Release-iphoneos/arm64/func.o
  #   /path/to/somewhere/Debug-iphonesimulator/x86_64/func.o

When any Xcode variable is used, CMake is not able to fully evaluate the
path(s) at configure time.  One consequence of this is that the
configuration-specific :prop_tgt:`IMPORTED_OBJECTS_<CONFIG>` properties cannot
be used, since CMake cannot determine whether an object file exists at a
particular ``<CONFIG>`` location.  The ``IMPORTED_OBJECTS`` property must be
used for these situations and the configuration-specific aspects of the path
should be handled by the ``$(CONFIGURATION)`` Xcode variable.
