CPack WIX Generator
-------------------

Use the `WiX Toolset`_ to produce a Windows Installer ``.msi`` database.

.. _`WiX Toolset`: https://www.firegiant.com/wixtoolset/

.. versionadded:: 3.7
  The :variable:`CPACK_COMPONENT_<compName>_DISABLED` variable is now
  supported.

WiX Toolsets
^^^^^^^^^^^^

CPack selects one of the following variants of the WiX Toolset
based on the :variable:`CPACK_WIX_VERSION` variable:

* `WiX .NET Tools`_
* `WiX Toolset v3`_

WiX .NET Tools
""""""""""""""

Packaging is performed using the following tools:

``wix build``
  Build WiX source files directly into a Windows Installer ``.msi`` database.

  Invocations may be customized using tool-specific variables:

  * :variable:`CPACK_WIX_BUILD_EXTENSIONS <CPACK_WIX_<TOOL>_EXTENSIONS>`
  * :variable:`CPACK_WIX_BUILD_EXTRA_FLAGS <CPACK_WIX_<TOOL>_EXTRA_FLAGS>`

WiX extensions must be named with the form ``WixToolset.<Name>.wixext``.

CPack expects the ``wix`` .NET tool to be available for command-line use
with any required WiX extensions already installed.  Be sure the ``wix``
version is compatible with :variable:`CPACK_WIX_VERSION`, and that WiX
extension versions match the ``wix`` tool version.  For example:

1. Install the ``wix`` command-line tool using ``dotnet``.

  To install ``wix`` globally for the current user:

  .. code-block:: bat

    dotnet tool install --global wix --version 4.0.4

  This places ``wix.exe`` in ``%USERPROFILE%\.dotnet\tools`` and adds
  the directory to the current user's ``PATH`` environment variable.

  Or, to install ``wix`` in a specific path, e.g., in ``c:\WiX``:

  .. code-block:: bat

    dotnet tool install --tool-path c:\WiX wix --version 4.0.4

  This places ``wix.exe`` in ``c:\WiX``, but does *not* add it to the
  current user's ``PATH`` environment variable.  The ``WIX`` environment
  variable may be set to tell CPack where to find the tool,
  e.g., ``set WIX=c:\WiX``.

2. Add the WiX ``UI`` extension, needed by CPack's default WiX template:

  .. code-block:: bat

    wix extension add --global WixToolset.UI.wixext/4.0.4

  Extensions added globally are stored in ``%USERPROFILE%\.wix``, or if the
  ``WIX_EXTENSIONS`` environment variable is set, in ``%WIX_EXTENSIONS%\.wix``.

WiX Toolset v3
""""""""""""""

Packaging is performed using the following tools:

``candle``
  Compiles WiX source files into ``.wixobj`` files.

  Invocations may be customized using tool-specific variables:

  * :variable:`CPACK_WIX_CANDLE_EXTENSIONS <CPACK_WIX_<TOOL>_EXTENSIONS>`
  * :variable:`CPACK_WIX_CANDLE_EXTRA_FLAGS <CPACK_WIX_<TOOL>_EXTRA_FLAGS>`

``light``
  Links ``.wixobj`` files into a Windows Installer ``.msi`` database.

  Invocations may be customized using tool-specific variables:

  * :variable:`CPACK_WIX_LIGHT_EXTENSIONS <CPACK_WIX_<TOOL>_EXTENSIONS>`
  * :variable:`CPACK_WIX_LIGHT_EXTRA_FLAGS <CPACK_WIX_<TOOL>_EXTRA_FLAGS>`

CPack invokes both tools as needed.  Intermediate ``.wixobj`` files
are considered implementation details.

WiX extensions must be named with the form ``Wix<Name>Extension``.

CPack expects the above tools to be available for command-line
use via the ``PATH``.  Or, if the ``WIX`` environment variable is set,
CPack looks for the tools in ``%WIX%`` and ``%WIX%\bin``.

Variables specific to CPack WIX generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following variables are specific to the installers built on
Windows using WiX.

.. variable:: CPACK_WIX_VERSION

 .. versionadded:: 3.30

 Specify the version of WiX Toolset for which the configuration
 is written.  The value must be one of

 ``4``
   Package using `WiX .NET Tools`_.

 ``3``
   Package using `WiX Toolset v3`_.  This is the default.

.. variable:: CPACK_WIX_UPGRADE_GUID

 Upgrade GUID (``Product/@UpgradeCode``)

 Will be automatically generated unless explicitly provided.

 It should be explicitly set to a constant generated globally unique
 identifier (GUID) to allow your installers to replace existing
 installations that use the same GUID.

 You may for example explicitly set this variable in your
 CMakeLists.txt to the value that has been generated per default.  You
 should not use GUIDs that you did not generate yourself or which may
 belong to other projects.

 A GUID shall have the following fixed length syntax::

  XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX

 (each X represents an uppercase hexadecimal digit)

.. variable:: CPACK_WIX_PRODUCT_GUID

 Product GUID (``Product/@Id``)

 Will be automatically generated unless explicitly provided.

 If explicitly provided this will set the Product Id of your installer.

 The installer will abort if it detects a pre-existing installation that
 uses the same GUID.

 The GUID shall use the syntax described for CPACK_WIX_UPGRADE_GUID.

.. variable:: CPACK_WIX_LICENSE_RTF

 RTF License File

 If CPACK_RESOURCE_FILE_LICENSE has an .rtf extension it is used as-is.

 If CPACK_RESOURCE_FILE_LICENSE has an .txt extension it is implicitly
 converted to RTF by the WIX Generator.
 The expected encoding of the .txt file is UTF-8.

 With CPACK_WIX_LICENSE_RTF you can override the license file used by the
 WIX Generator in case CPACK_RESOURCE_FILE_LICENSE is in an unsupported
 format or the .txt -> .rtf conversion does not work as expected.

.. variable:: CPACK_WIX_PRODUCT_ICON

 The Icon shown next to the program name in Add/Remove programs.

 If set, this icon is used in place of the default icon.

.. variable:: CPACK_WIX_UI_REF

 Specify the WiX ``UI`` extension's dialog set:

 * With `WiX .NET Tools`_, this is the Id of the
   ``<ui:WixUI>`` element in the default WiX template.

 * With `WiX Toolset v3`_, this is the Id of the
   ``<UIRef>`` element in the default WiX template.

 The default is ``WixUI_InstallDir`` in case no CPack components have
 been defined and ``WixUI_FeatureTree`` otherwise.

.. variable:: CPACK_WIX_UI_BANNER

 The bitmap will appear at the top of all installer pages other than the
 welcome and completion dialogs.

 If set, this image will replace the default banner image.

 This image must be 493 by 58 pixels.

.. variable:: CPACK_WIX_UI_DIALOG

 Background bitmap used on the welcome and completion dialogs.

 If this variable is set, the installer will replace the default dialog
 image.

 This image must be 493 by 312 pixels.

.. variable:: CPACK_WIX_PROGRAM_MENU_FOLDER

 Start menu folder name for launcher.

 If this variable is not set, it will be initialized with CPACK_PACKAGE_NAME

 .. versionadded:: 3.16
  If this variable is set to ``.``, then application shortcuts will be
  created directly in the start menu and the uninstaller shortcut will be
  omitted.

.. variable:: CPACK_WIX_CULTURES

 Language(s) of the installer

 Languages are compiled into the Wix ``UI`` extension library.  To use them,
 simply provide the name of the culture.  If you specify more than one
 culture identifier in a comma or semicolon delimited list, the first one
 that is found will be used.  You can find a list of supported languages at:
 https://docs.firegiant.com/wix3/wixui/wixui_localization/

.. variable:: CPACK_WIX_TEMPLATE

 Template file for WiX generation

 If this variable is set, the specified template will be used to generate
 the WiX wxs file.  This should be used if further customization of the
 output is required. The template contents will override the effect of most
 ``CPACK_WIX_`` variables.

 If this variable is not set, the default MSI template included with CMake
 will be used.

.. variable:: CPACK_WIX_PATCH_FILE

 Optional list of XML files with fragments to be inserted into
 generated WiX sources.

 .. versionadded:: 3.5
  Support listing multiple patch files.

 This optional variable can be used to specify an XML file that the
 WIX generator will use to inject fragments into its generated
 source files.

 Patch files understood by the CPack WIX generator
 roughly follow this RELAX NG compact schema:

 .. code-block:: none

    start = CPackWiXPatch

    CPackWiXPatch = element CPackWiXPatch { CPackWiXFragment* }

    CPackWiXFragment = element CPackWiXFragment
    {
        attribute Id { string },
        fragmentContent*
    }

    fragmentContent = element * - CPackWiXFragment
    {
        (attribute * { text } | text | fragmentContent)*
    }

 Currently fragments can be injected into most
 Component, File, Directory and Feature elements.

 .. versionadded:: 3.3
  The following additional special Ids can be used:

  * ``#PRODUCT`` for the ``<Product>`` element.
  * ``#PRODUCTFEATURE`` for the root ``<Feature>`` element.

 .. versionadded:: 3.7
  Support patching arbitrary ``<Feature>`` elements.

 .. versionadded:: 3.9
  Allow setting additional attributes.

 The following example illustrates how this works.

 Given that the WIX generator creates the following XML element:

 .. code-block:: xml

    <Component Id="CM_CP_applications.bin.my_libapp.exe" Guid="*"/>

 The following XML patch file may be used to inject an Environment element
 into it:

 .. code-block:: xml

    <CPackWiXPatch>
      <CPackWiXFragment Id="CM_CP_applications.bin.my_libapp.exe">
        <Environment Id="MyEnvironment" Action="set"
          Name="MyVariableName" Value="MyVariableValue"/>
      </CPackWiXFragment>
    </CPackWiXPatch>

.. variable:: CPACK_WIX_EXTRA_SOURCES

 Extra WiX source files

 This variable provides an optional list of extra WiX source files (``.wxs``)
 that should be compiled and linked.  The paths must be absolute.

.. variable:: CPACK_WIX_EXTRA_OBJECTS

 Extra WiX object files or libraries to use with `WiX Toolset v3`_.

 This variable provides an optional list of extra WiX object (``.wixobj``)
 and/or WiX library (``.wixlib``) files.  The paths must be absolute.

.. variable:: CPACK_WIX_EXTENSIONS

 Specify a list of additional extensions for WiX tools.
 See `WiX Toolsets`_ for extension naming patterns.

.. variable:: CPACK_WIX_<TOOL>_EXTENSIONS

 Specify a list of additional extensions for a specific WiX tool.
 See `WiX Toolsets`_ for possible ``<TOOL>`` names.

.. variable:: CPACK_WIX_<TOOL>_EXTRA_FLAGS

 Specify a list of additional command-line flags for a specific WiX tool.
 See `WiX Toolsets`_ for possible ``<TOOL>`` names.

 Use it at your own risk.
 Future versions of CPack may generate flags which may be in conflict
 with your own flags.

.. variable:: CPACK_WIX_CMAKE_PACKAGE_REGISTRY

 If this variable is set the generated installer will create
 an entry in the windows registry key
 ``HKEY_LOCAL_MACHINE\Software\Kitware\CMake\Packages\<PackageName>``
 The value for ``<PackageName>`` is provided by this variable.

 Assuming you also install a CMake configuration file this will
 allow other CMake projects to find your package with
 the :command:`find_package` command.

.. variable:: CPACK_WIX_PROPERTY_<PROPERTY>

 .. versionadded:: 3.1

 This variable can be used to provide a value for
 the Windows Installer property ``<PROPERTY>``

 The following list contains some example properties that can be used to
 customize information under
 "Programs and Features" (also known as "Add or Remove Programs")

 * ARPCOMMENTS - Comments
 * ARPHELPLINK - Help and support information URL
 * ARPURLINFOABOUT - General information URL
 * ARPURLUPDATEINFO - Update information URL
 * ARPHELPTELEPHONE - Help and support telephone number
 * ARPSIZE - Size (in kilobytes) of the application

.. variable:: CPACK_WIX_ROOT_FEATURE_TITLE

 .. versionadded:: 3.7

 Sets the name of the root install feature in the WIX installer. Same as
 CPACK_COMPONENT_<compName>_DISPLAY_NAME for components.

.. variable:: CPACK_WIX_ROOT_FEATURE_DESCRIPTION

 .. versionadded:: 3.7

 Sets the description of the root install feature in the WIX installer. Same as
 CPACK_COMPONENT_<compName>_DESCRIPTION for components.

.. variable:: CPACK_WIX_SKIP_PROGRAM_FOLDER

 .. versionadded:: 3.7

 If this variable is set to true, the default install location
 of the generated package will be CPACK_PACKAGE_INSTALL_DIRECTORY directly.
 The install location will not be located relatively below
 ProgramFiles or ProgramFiles64.

  .. note::
    Installers created with this feature do not take differences
    between the system on which the installer is created
    and the system on which the installer might be used into account.

    It is therefore possible that the installer e.g. might try to install
    onto a drive that is unavailable or unintended or a path that does not
    follow the localization or convention of the system on which the
    installation is performed.

.. variable:: CPACK_WIX_ROOT_FOLDER_ID

 .. versionadded:: 3.9

 This variable allows specification of a custom root folder ID.
 The generator specific ``<64>`` token can be used for
 folder IDs that come in 32-bit and 64-bit variants.
 In 32-bit builds the token will expand empty while in 64-bit builds
 it will expand to ``64``.

 When unset generated installers will default installing to
 ``ProgramFiles<64>Folder``.

.. variable:: CPACK_WIX_ROOT

 This variable can optionally be set to the root directory
 of a custom WiX Toolset installation.

 When unspecified CPack will try to locate a WiX Toolset
 installation via the ``WIX`` environment variable instead.

.. variable:: CPACK_WIX_CUSTOM_XMLNS

 .. versionadded:: 3.19

 This variable provides a list of custom namespace declarations that are necessary
 for using WiX extensions. Each declaration should be in the form name=url, where
 name is the plain namespace without the usual xmlns: prefix and url is an unquoted
 namespace url. A list of commonly known WiX schemata can be found here:
 https://docs.firegiant.com/wix3/xsd/

.. variable:: CPACK_WIX_SKIP_WIX_UI_EXTENSION

 .. versionadded:: 3.23

 If this variable is set to true, the default inclusion of the WiX ``UI``
 extension is skipped, i.e., the ``-ext WixUIExtension`` or
 ``-ext WixToolset.UI.wixext`` flag is not passed to WiX tools.

.. variable:: CPACK_WIX_ARCHITECTURE

 .. versionadded:: 3.24

 This variable can be optionally set to specify the target architecture
 of the installer. May for example be set to ``x64`` or ``arm64``.

 When unspecified, CPack will default to ``x64`` or ``x86``.

.. variable:: CPACK_WIX_INSTALL_SCOPE

 .. versionadded:: 3.29

 This variable can be optionally set to specify the ``InstallScope``
 of the installer:

 ``perMachine``
   Create an installer that installs for all users and requires
   administrative privileges.  Start menu entries created by the
   installer are visible to all users.

   This is the default.  See policy :policy:`CMP0172`.

 ``perUser``
   Not yet supported. This is reserved for future use.

 ``NONE``
   Create an installer without any ``InstallScope`` attribute.

   This is supported only if :variable:`CPACK_WIX_VERSION` is not set,
   or is set to ``3``.

   .. deprecated:: 3.29

     This value is only for compatibility with the inconsistent behavior used
     by CPack 3.28 and older.  The resulting installer requires administrative
     privileges and installs into the system-wide ``ProgramFiles`` directory,
     but the start menu entry and uninstaller registration are created only
     for the current user.

   .. warning::

     An installation performed by an installer created without any
     ``InstallScope`` cannot be cleanly updated or replaced by an
     installer with an ``InstallScope``.  In order to transition
     a project's installers from ``NONE`` to ``perMachine``, the
     latter installer should be distributed with instructions to
     first manually uninstall any older version.

 See https://docs.firegiant.com/wix3/xsd/wix/package/

.. variable:: CPACK_WIX_CAB_PER_COMPONENT

 .. versionadded:: 4.2

 If this variable is set to true one ``.cab`` file per component is created.
 The default is to create a single ``.cab`` file for all files in the installer.

 WiX creates ``.cab`` files in parallel so multiple ``.cab`` files may be
 desirable for faster packaging.
