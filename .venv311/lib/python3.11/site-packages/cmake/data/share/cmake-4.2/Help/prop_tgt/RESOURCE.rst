RESOURCE
--------

Specify resource files in a :prop_tgt:`FRAMEWORK` or :prop_tgt:`BUNDLE`.

Target marked with the :prop_tgt:`FRAMEWORK` or :prop_tgt:`BUNDLE` property
generate framework or application bundle (both macOS and iOS is supported)
or normal shared libraries on other platforms.
This property may be set to a list of files to be placed in the corresponding
directory (eg. ``Resources`` directory for macOS) inside the bundle.
On non-Apple platforms these files may be installed using the ``RESOURCE``
option to the :command:`install(TARGETS)` command.

Following example of Application Bundle:

.. code-block:: cmake

  add_executable(ExecutableTarget
    addDemo.c
    resourcefile.txt
    appresourcedir/appres.txt)

  target_link_libraries(ExecutableTarget heymath mul)

  set(RESOURCE_FILES
    resourcefile.txt
    appresourcedir/appres.txt)

  set_target_properties(ExecutableTarget PROPERTIES
    MACOSX_BUNDLE TRUE
    MACOSX_FRAMEWORK_IDENTIFIER org.cmake.ExecutableTarget
    RESOURCE "${RESOURCE_FILES}")

will produce flat structure for iOS systems::

  ExecutableTarget.app
    appres.txt
    ExecutableTarget
    Info.plist
    resourcefile.txt

For macOS systems it will produce following directory structure::

  ExecutableTarget.app/
    Contents
      Info.plist
      MacOS
        ExecutableTarget
      Resources
        appres.txt
        resourcefile.txt

For Linux, such CMake script produce following files::

  ExecutableTarget
  Resources
    appres.txt
    resourcefile.txt
