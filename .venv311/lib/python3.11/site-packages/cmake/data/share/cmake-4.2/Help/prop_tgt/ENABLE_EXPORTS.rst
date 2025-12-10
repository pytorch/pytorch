ENABLE_EXPORTS
--------------

Specify whether an executable or a shared library exports symbols.

Normally an executable does not export any symbols because it is the
final program.  It is possible for an executable to export symbols to
be used by loadable modules.  When this property is set to true CMake
will allow other targets to "link" to the executable with the
:command:`target_link_libraries` command.  On all platforms a target-level
dependency on the executable is created for targets that link to it.
Handling of the executable on the link lines of the loadable modules
varies by platform:

* On Windows-based systems (including Cygwin) an "import library" is
  created along with the executable to list the exported symbols.
  Loadable modules link to the import library to get the symbols.

* On macOS, loadable modules link to the executable itself using the
  ``-bundle_loader`` flag.

* On AIX, a linker "import file" is created along with the executable
  to list the exported symbols for import when linking other targets.
  Loadable modules link to the import file to get the symbols.

* On other platforms, loadable modules are simply linked without
  referencing the executable since the dynamic loader will
  automatically bind symbols when the module is loaded.

This property is initialized by the value of the
:variable:`CMAKE_EXECUTABLE_ENABLE_EXPORTS` variable, if it is set when an
executable target is created.  If :variable:`CMAKE_EXECUTABLE_ENABLE_EXPORTS`
is not set, the :variable:`CMAKE_ENABLE_EXPORTS` variable is used to initialize
the property instead for backward compatibility reasons.
See below for alternative initialization behavior for shared library targets.

.. versionadded:: 3.27
  To link with a shared library on macOS, or to a shared framework on any Apple
  platform, a linker import file can be used instead of the actual shared
  library. These linker import files are also known as text-based stubs, and
  they have a ``.tbd`` file extension.

  The generation of these linker import files, as well as their consumption, is
  controlled by this property. When this property is set to true on a shared
  library target, CMake will generate a ``.tbd`` file for the library.
  Other targets that link to the shared library target will then use this
  ``.tbd`` file when linking rather than linking to the shared library binary.

  .. note::

    For backward compatibility reasons, this property will be ignored if the
    :prop_tgt:`XCODE_ATTRIBUTE_GENERATE_TEXT_BASED_STUBS <XCODE_ATTRIBUTE_<an-attribute>>`
    target property or the
    :variable:`CMAKE_XCODE_ATTRIBUTE_GENERATE_TEXT_BASED_STUBS <CMAKE_XCODE_ATTRIBUTE_<an-attribute>>`
    variable is set to false.

  For shared library targets, this property is initialized by the value of the
  :variable:`CMAKE_SHARED_LIBRARY_ENABLE_EXPORTS` variable, if it is set when
  the target is created.
