CPack NSIS Generator
--------------------

CPack Nullsoft Scriptable Install System (NSIS) generator specific options.

.. versionchanged:: 3.22
 The NSIS generator requires NSIS 3.03 or newer.

Variables specific to CPack NSIS generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following variables are specific to the graphical installers built
on Windows Nullsoft Scriptable Install System.

.. variable:: CPACK_NSIS_INSTALL_ROOT

 The default installation directory presented to the end user by the NSIS
 installer is under this root dir. The full directory presented to the end
 user is: ``${CPACK_NSIS_INSTALL_ROOT}/${CPACK_PACKAGE_INSTALL_DIRECTORY}``

.. variable:: CPACK_NSIS_MUI_ICON

 An icon filename.  The name of a ``*.ico`` file used as the main icon for the
 generated install program.

.. variable:: CPACK_NSIS_MUI_UNIICON

 An icon filename.  The name of a ``*.ico`` file used as the main icon for the
 generated uninstall program.

.. variable:: CPACK_NSIS_INSTALLER_MUI_ICON_CODE

 undocumented.

.. variable:: CPACK_NSIS_MUI_WELCOMEFINISHPAGE_BITMAP

 .. versionadded:: 3.5

 The filename of a bitmap to use as the NSIS ``MUI_WELCOMEFINISHPAGE_BITMAP``.

.. variable:: CPACK_NSIS_MUI_UNWELCOMEFINISHPAGE_BITMAP

 .. versionadded:: 3.5

 The filename of a bitmap to use as the NSIS ``MUI_UNWELCOMEFINISHPAGE_BITMAP``.

.. variable:: CPACK_NSIS_EXTRA_PREINSTALL_COMMANDS

 Extra NSIS commands that will be added to the beginning of the install
 Section, before your install tree is available on the target system.

.. variable:: CPACK_NSIS_EXTRA_INSTALL_COMMANDS

 Extra NSIS commands that will be added to the end of the install Section,
 after your install tree is available on the target system.

.. variable:: CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS

 Extra NSIS commands that will be added to the uninstall Section, before
 your install tree is removed from the target system.

.. variable:: CPACK_NSIS_COMPRESSOR

 The arguments that will be passed to the NSIS ``SetCompressor`` command.

.. variable:: CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL

 Ask about uninstalling previous versions first.  If this is set to ``ON``,
 then an installer will look for previous installed versions and if one is
 found, ask the user whether to uninstall it before proceeding with the
 install.

.. variable:: CPACK_NSIS_MODIFY_PATH

 Modify ``PATH`` toggle.  If this is set to ``ON``, then an extra page will appear
 in the installer that will allow the user to choose whether the program
 directory should be added to the system ``PATH`` variable.

.. variable:: CPACK_NSIS_DISPLAY_NAME

 The display name string that appears in the Windows *Apps & features*
 in *Control Panel*

.. variable:: CPACK_NSIS_PACKAGE_NAME

 The title displayed at the top of the installer.

.. variable:: CPACK_NSIS_INSTALLED_ICON_NAME

 A path to the executable that contains the installer icon.

.. variable:: CPACK_NSIS_HELP_LINK

 URL to a web site providing assistance in installing your application.

.. variable:: CPACK_NSIS_URL_INFO_ABOUT

 URL to a web site providing more information about your application.

.. variable:: CPACK_NSIS_CONTACT

 Contact information for questions and comments about the installation
 process.

.. variable:: CPACK_NSIS_<compName>_INSTALL_DIRECTORY

 .. versionadded:: 3.7

 Custom install directory for the specified component ``<compName>`` instead
 of ``$INSTDIR``.

.. variable:: CPACK_NSIS_CREATE_ICONS_EXTRA

 Additional NSIS commands for creating *Start Menu* shortcuts.

.. variable:: CPACK_NSIS_DELETE_ICONS_EXTRA

 Additional NSIS commands to uninstall *Start Menu* shortcuts.

.. variable:: CPACK_NSIS_EXECUTABLES_DIRECTORY

 Creating NSIS *Start Menu* links assumes that they are in ``bin`` unless this
 variable is set.  For example, you would set this to ``exec`` if your
 executables are in an exec directory.

.. variable:: CPACK_NSIS_MUI_FINISHPAGE_RUN

 Specify an executable to add an option to run on the finish page of the
 NSIS installer.

.. variable:: CPACK_NSIS_MENU_LINKS

 Specify links in ``[application]`` menu.  This should contain a list of pair
 ``link`` ``link name``. The link may be a URL or a path relative to
 installation prefix.  Like:

 .. code-block:: cmake

    set(CPACK_NSIS_MENU_LINKS
      "doc/cmake-@CMake_VERSION_MAJOR@.@CMake_VERSION_MINOR@/cmake.html"
      "CMake Help" "https://cmake.org" "CMake Web Site")

.. variable:: CPACK_NSIS_UNINSTALL_NAME

 .. versionadded:: 3.17

 Specify the name of the program to uninstall the version.
 Default is ``Uninstall``.

.. variable:: CPACK_NSIS_WELCOME_TITLE

  .. versionadded:: 3.17

  The title to display on the top of the page for the welcome page.

.. variable:: CPACK_NSIS_WELCOME_TITLE_3LINES

 .. versionadded:: 3.17

 Display the title in the welcome page on 3 lines instead of 2.

.. variable:: CPACK_NSIS_FINISH_TITLE

 .. versionadded:: 3.17

 The title to display on the top of the page for the finish page.

.. variable:: CPACK_NSIS_FINISH_TITLE_3LINES

 .. versionadded:: 3.17

 Display the title in the finish page on 3 lines instead of 2.

.. variable:: CPACK_NSIS_MUI_HEADERIMAGE

 .. versionadded:: 3.17

 The image to display on the header of installers pages.

.. variable:: CPACK_NSIS_MANIFEST_DPI_AWARE

 .. versionadded:: 3.18

 If set, declares that the installer is DPI-aware.

.. variable:: CPACK_NSIS_BRANDING_TEXT

 .. versionadded:: 3.20

 If set, updates the text at the bottom of the install window.
 To set the string to blank, use a space (" ").

.. variable:: CPACK_NSIS_BRANDING_TEXT_TRIM_POSITION

 .. versionadded:: 3.20

 If set, trim down the size of the control to the size of the branding text string.
 Allowed values for this variable are ``LEFT``, ``CENTER`` or ``RIGHT``.
 If not specified, the default behavior is ``LEFT``.

.. variable:: CPACK_NSIS_EXECUTABLE

 .. versionadded:: 3.21

 If set, specify the name of the NSIS executable. Default is ``makensis``.

.. variable:: CPACK_NSIS_IGNORE_LICENSE_PAGE

 .. versionadded:: 3.22

 If set, do not display the page containing the license during installation.

.. variable:: CPACK_NSIS_EXECUTABLE_PRE_ARGUMENTS

 .. versionadded:: 3.25

 This variable is a :ref:`semicolon-separated list <CMake Language Lists>` of
 arguments to prepend to the nsis script to run.
 If the arguments do not start with a ``/`` or a ``-``, it will add one
 automatically to the corresponding arguments.
 The command that will be run is::

    makensis.exe <preArgs>... "nsisFileName.nsi" <postArgs>...

 where ``<preArgs>...`` is constructed from ``CPACK_NSIS_EXECUTABLE_PRE_ARGUMENTS``
 and ``<postArgs>...``  is constructed from ``CPACK_NSIS_EXECUTABLE_POST_ARGUMENTS``.


.. variable:: CPACK_NSIS_EXECUTABLE_POST_ARGUMENTS

 .. versionadded:: 3.25

 This variable is a :ref:`semicolon-separated list <CMake Language Lists>` of
 arguments to append to the nsis script to run.
 If the arguments do not start with a ``/`` or a ``-``, it will add one
 automatically to the corresponding arguments.
 The command that will be run is::

    makensis.exe <preArgs>... "nsisFileName.nsi" <postArgs>...

 where ``<preArgs>...`` is constructed from ``CPACK_NSIS_EXECUTABLE_PRE_ARGUMENTS``
 and ``<postArgs>...``  is constructed from ``CPACK_NSIS_EXECUTABLE_POST_ARGUMENTS``.

.. variable:: CPACK_NSIS_CRC_CHECK

 .. versionadded:: 4.2

 Specifies whether or not the installer will perform a CRC on itself before
 allowing an install.
 Allowed values for this variable are ``on``, ``off``, and ``force``.
 If not specified, the default behavior is ``on``.
