CPack Inno Setup Generator
--------------------------

.. versionadded:: 3.27

Inno Setup is a free installer for Windows programs by Jordan Russell and
Martijn Laan (https://jrsoftware.org/isinfo.php).

This documentation explains Inno Setup generator specific options.

The generator provides a lot of options like components. Unfortunately, not
all features (e.g. component dependencies) are currently supported by
Inno Setup and they're ignored by the generator for now.

CPack requires Inno Setup 6 or greater.

.. versionadded:: 3.30
  The generator is now available on non-Windows hosts,
  but requires Wine to run the Inno Setup tools.

Variables specific to CPack Inno Setup generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use the following variables to change the behavior of the CPack
``INNOSETUP`` generator:


General
"""""""

None of the following variables is required to be set for the Inno Setup
generator to work. If a variable is marked as mandatory below but not set,
its default value is taken.

The variables can also contain Inno Setup constants like ``{app}``. Please
refer to the documentation of Inno Setup for more information.

If you're asked to provide the path to any file, you can always give an
absolute path or in most cases the relative path from the top-level directory
where all files being installed by an :command:`install` instruction reside.

CPack tries to escape quotes and other special characters for you. However,
using special characters could cause problems.

The following variable simplifies the usage of Inno Setup in CMake:

.. variable:: CPACK_INNOSETUP_USE_CMAKE_BOOL_FORMAT

 Inno Setup only uses ``yes`` or ``no`` as boolean formats meanwhile CMake
 uses a lot of alternative formats like ``ON`` or ``OFF``. Having this option
 turned on enables an automatic conversion.

 Consider the following example:

 .. code-block:: cmake

  set(CMAKE_INNOSETUP_SETUP_AllowNoIcons OFF)

 If this option is turned on, the following line will be created in the output
 script: ``AllowNoIcons=no``.
 Else, the following erroneous line will be created: ``AllowNoIcons=OFF``

 The conversion is enabled in every Inno Setup specific variable.

 :Mandatory: Yes
 :Default: ``ON``


Setup Specific Variables
""""""""""""""""""""""""

.. variable:: CPACK_INNOSETUP_ARCHITECTURE

 One of ``x86``, ``x64``, ``arm64`` or ``ia64``. This variable specifies the
 target architecture of the installer. This also affects the Program Files
 folder or registry keys being used.

 CPack tries to determine the correct value with a try compile (see
 :variable:`CMAKE_SIZEOF_VOID_P`), but this option can be manually specified
 too (especially when using ``ia64`` or cross-platform compilation).

 :Mandatory: Yes
 :Default: Either ``x86`` or ``x64`` depending on the results of the try-compile

.. variable:: CPACK_INNOSETUP_INSTALL_ROOT

 If you don't want the installer to create the installation directory under
 Program Files, you've to specify the installation root here.

 The full directory of the installation will be:
 ``${CPACK_INNOSETUP_INSTALL_ROOT}/${CPACK_PACKAGE_INSTALL_DIRECTORY}``.

 :Mandatory: Yes
 :Default: ``{autopf}``

.. variable:: CPACK_INNOSETUP_ALLOW_CUSTOM_DIRECTORY

 If turned on, the installer allows the user to change the installation
 directory providing an extra wizard page.

 :Mandatory: Yes
 :Default: ``ON``

.. variable:: CPACK_INNOSETUP_PROGRAM_MENU_FOLDER

 The initial name of the start menu folder being created.

 If this variable is set to ``.``, then no separate folder is created,
 application shortcuts will appear in the top-level start menu folder.

 :Mandatory: Yes
 :Default: The value of :variable:`CPACK_PACKAGE_NAME`

.. variable:: CPACK_INNOSETUP_LANGUAGES

 A :ref:`semicolon-separated list <CMake Language Lists>` of languages you want
 Inno Setup to include.

 Currently available: ``armenian``, ``brazilianPortuguese``, ``bulgarian``,
 ``catalan``, ``corsican``, ``czech``, ``danish``, ``dutch``, ``english``,
 ``finnish``, ``french``, ``german``, ``hebrew``, ``icelandic``, ``italian``,
 ``japanese``, ``norwegian``, ``polish``, ``portuguese``, ``russian``,
 ``slovak``, ``slovenian``, ``spanish``, ``turkish`` and ``ukrainian``.
 This list might differ depending on the version of Inno Setup.

 :Mandatory: Yes
 :Default: ``english``

.. variable:: CPACK_INNOSETUP_IGNORE_LICENSE_PAGE

 If you don't specify a license file using
 :variable:`CPACK_RESOURCE_FILE_LICENSE`, CPack uses a file for demonstration
 purposes. If you want the installer to ignore license files at all, you can
 enable this option.

 :Mandatory: Yes
 :Default: ``OFF``

.. variable:: CPACK_INNOSETUP_IGNORE_README_PAGE

 If you don't specify a readme file using
 :variable:`CPACK_RESOURCE_FILE_README`, CPack uses a file for demonstration
 purposes. If you want the installer to ignore readme files at all, you can
 enable this option. Make sure the option is disabled when using
 a custom readme file.

 :Mandatory: Yes
 :Default: ``ON``

.. variable:: CPACK_INNOSETUP_PASSWORD

 Enables password protection and file encryption with the given password.

 :Mandatory: No

.. variable:: CPACK_INNOSETUP_USE_MODERN_WIZARD

 Enables the modern look and feel provided by Inno Setup. If this option is
 turned off, the classic style is used instead. Images and icon files are
 also affected.

 :Mandatory: Yes
 :Default: ``OFF`` because of compatibility reasons

.. variable:: CPACK_INNOSETUP_ICON_FILE

 The path to a custom installer ``.ico`` file.

 Use :variable:`CPACK_PACKAGE_ICON` to customize the bitmap file being shown
 in the wizard.

 :Mandatory: No

.. variable:: CPACK_INNOSETUP_SETUP_<directive>

 This group allows adapting any of the ``[Setup]`` section directives provided
 by Inno Setup where ``directive`` is its name.

 Here are some examples:

 .. code-block:: cmake

  set(CPACK_INNOSETUP_SETUP_WizardSmallImageFile "my_bitmap.bmp")
  set(CPACK_INNOSETUP_SETUP_AllowNoIcons OFF) # This requires CPACK_INNOSETUP_USE_CMAKE_BOOL_FORMAT to be on

 All of these variables have higher priority than the others.
 Consider the following example:

 .. code-block:: cmake

  set(CPACK_INNOSETUP_SETUP_Password "admin")
  set(CPACK_INNOSETUP_PASSWORD "secret")

 The password will be ``admin`` at the end because ``CPACK_INNOSETUP_PASSWORD``
 has less priority than ``CPACK_INNOSETUP_SETUP_Password``.

 :Mandatory: No


File Specific Variables
"""""""""""""""""""""""

Although all files being installed by an :command:`install` instruction are
automatically processed and added to the installer, there are some variables
to customize the installation process.

Before using executables (only ``.exe`` or ``.com``) in shortcuts
(e.g. :variable:`CPACK_CREATE_DESKTOP_LINKS`) or ``[Run]`` entries, you've to
add the raw file name (without path and extension) to
:variable:`CPACK_PACKAGE_EXECUTABLES` and create a start menu shortcut
for them.

If you have two files with the same raw name (e.g. ``a/executable.exe`` and
``b/executable.com``), an entry in the section is created twice. This will
result in undefined behavior and is not recommended.

.. variable:: CPACK_INNOSETUP_CUSTOM_INSTALL_INSTRUCTIONS

 This variable should contain a
 :ref:`semicolon-separated list <CMake Language Lists>` of pairs ``path``,
 ``instruction`` and can be used to customize the install command being
 automatically created for each file or directory.

 CPack creates the following Inno Setup instruction for every file...

 .. code-block::

  Source: "absolute\path\to\my_file.txt"; DestDir: "{app}"; Flags: ignoreversion

 ...and the following line for every directory:

 .. code-block::

  Name: "{app}\my_folder"

 You might want to change the destination directory or the flags of
 ``my_file.txt``. Since we can also provide a relative path, the line you'd
 like to have, is the following:

 .. code-block::

  Source: "my_file.txt"; DestDir: "{userdocs}"; Flags: ignoreversion uninsneveruninstall

 You would do this by using ``my_file.txt`` as ``path`` and
 ``Source: "my_file.txt"; DestDir: "{userdocs}"; Flags: ignoreversion uninsneveruninstall``
 as ``instruction``.

 You've to take care of the `escaping problem <https://cmake.org/cmake/help/book/mastering-cmake/chapter/Packaging%20With%20CPack.html#adding-custom-cpack-options>`_.
 So the CMake command would be:

 .. code-block:: cmake

  set(CPACK_INNOSETUP_CUSTOM_INSTALL_INSTRUCTIONS "my_file.txt;Source: \\\"my_file.txt\\\"\\; DestDir: \\\"{userdocs}\\\"\\; Flags: ignoreversion uninsneveruninstall")

 To improve readability, you should go around the escaping problem by using
 :variable:`CPACK_VERBATIM_VARIABLES` or by placing the instruction into a
 separate CPack project config file.

 If you customize the install instruction of a specific file, you lose the
 connection to its component. To go around, manually add
 ``Components: <component>``. You also need to add its shortcuts and ``[Run]``
 entries by yourself in a custom section, since the executable won't be found
 anymore by :variable:`CPACK_PACKAGE_EXECUTABLES`.

 Here's another example (Note: You've to go around the escaping problem for
 the example to work):

 .. code-block:: cmake

  set(CPACK_INNOSETUP_CUSTOM_INSTALL_INSTRUCTIONS
      "component1/my_folder" "Name: \"{userdocs}\\my_folder\"\; Components: component1"
      "component2/my_folder2/my_file.txt" "Source: \"component2\\my_folder2\\my_file.txt\"\; DestDir: \"{app}\\my_folder2\\my_file.txt\"\; Flags: ignoreversion uninsneveruninstall\; Components: component2")

 :Mandatory: No

.. variable:: CPACK_INNOSETUP_MENU_LINKS

 This variable should contain a
 :ref:`semicolon-separated list <CMake Language Lists>` of pairs ``link``,
 ``link name`` and can be used to add shortcuts into the start menu folder
 beside those of the executables (see :variable:`CPACK_PACKAGE_EXECUTABLES`).
 While ``link name`` is the label, ``link`` can be a URL or a path relative to
 the installation directory.

 Here's an example:

 .. code-block:: cmake

  set(CPACK_INNOSETUP_MENU_LINKS
      "doc/cmake-@CMake_VERSION_MAJOR@.@CMake_VERSION_MINOR@/cmake.html"
      "CMake Help" "https://cmake.org" "CMake Web Site")

 :Mandatory: No

.. variable:: CPACK_INNOSETUP_CREATE_UNINSTALL_LINK

 If this option is turned on, a shortcut to the application's uninstaller is
 automatically added to the start menu folder.

 :Mandatory: Yes
 :Default: ``OFF``

.. variable:: CPACK_INNOSETUP_RUN_EXECUTABLES

 A :ref:`semicolon-separated list <CMake Language Lists>` of executables being
 specified in :variable:`CPACK_PACKAGE_EXECUTABLES` which the user can run
 when the installer finishes.

 They're internally added to the ``[Run]`` section.

 :Mandatory: No


Components Specific Variables
"""""""""""""""""""""""""""""

The generator supports components and also downloaded components. However,
there are some features of components that aren't supported yet (especially
component dependencies). These variables are ignored for now.

CPack will change a component's name in Inno Setup if it has a parent group
for technical reasons. Consider using ``group\component`` as component name in
Inno Setup scripts if you have the component ``component`` and its parent
group ``group``.

Here are some additional variables for components:

.. variable::  CPACK_INNOSETUP_<compName>_INSTALL_DIRECTORY

 If you don't want the component ``compName`` to be installed under ``{app}``,
 you've to specify its installation directory here.

 :Mandatory: No

.. variable:: CPACK_INNOSETUP_VERIFY_DOWNLOADS

 This option only affects downloaded components.

 If this option is turned on, the hashes of the downloaded archives are
 calculated during compile and
 download time. The installer will only proceed if they match.

 :Mandatory: Yes
 :Default: ``ON``


Compilation and Scripting Specific Variables
""""""""""""""""""""""""""""""""""""""""""""

.. variable:: CPACK_INNOSETUP_EXECUTABLE

 The filename of the Inno Setup Script Compiler command.

 :Mandatory: Yes
 :Default: ``ISCC``

.. variable:: CPACK_INNOSETUP_EXECUTABLE_ARGUMENTS

 A :ref:`semicolon-separated list <CMake Language Lists>` of extra
 command-line options for the Inno Setup Script Compiler command.

 For example: ``/Qp;/Smysigntool=$p``

 Take care of the `escaping problem <https://cmake.org/cmake/help/book/mastering-cmake/chapter/Packaging%20With%20CPack.html#adding-custom-cpack-options>`_.

 :Mandatory: No

.. variable:: CPACK_INNOSETUP_DEFINE_<macro>

 This group allows to add custom define directives as command-line options to
 the Inno Setup Preprocessor command. Each entry emulates a
 ``#define public <macro>`` directive. Its macro is accessible from anywhere
 (``public``), so it can also be used in extra script files.

 Macro names must not contain any special characters. Refer to the Inno Setup
 Preprocessor documentation for the detailed rules.

 Consider the following example:

 .. code-block:: cmake

  # The following line emulates: #define public MyMacro "Hello, World!"
  set(CPACK_INNOSETUP_DEFINE_MyMacro "Hello, World!")

 At this point, you can use ``MyMacro`` anywhere. For example in the following
 extra script:

 .. code-block::

  AppComments={#emit "'My Macro' has the value: " + MyMacro}

 Take care of the `escaping problem <https://cmake.org/cmake/help/book/mastering-cmake/chapter/Packaging%20With%20CPack.html#adding-custom-cpack-options>`_.

 :Mandatory: No

.. variable:: CPACK_INNOSETUP_EXTRA_SCRIPTS

 A :ref:`semicolon-separated list <CMake Language Lists>` of paths to
 additional ``.iss`` script files to be processed.

 They're internally included at the top of the output script file using a
 ``#include`` directive.

 You can add any section in your file to extend the installer (e.g. adding
 additional tasks or registry keys). Prefer using
 :variable:`CPACK_INNOSETUP_SETUP_<directive>` when extending the
 ``[Setup]`` section.

 :Mandatory: No

.. variable:: CPACK_INNOSETUP_CODE_FILES

 A :ref:`semicolon-separated list <CMake Language Lists>` of paths to
 additional Pascal files to be processed.

 This variable is actually the same as
 :variable:`CPACK_INNOSETUP_EXTRA_SCRIPTS`, except you don't have to
 add ``[Code]`` at the top of your file. Never change the current section in
 a code file. This will result in undefined behavior! Treat them as normal
 Pascal scripts instead.

 Code files are included at the very bottom of the output script.

 :Mandatory: No
