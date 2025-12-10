CPack IFW Generator
-------------------

.. versionadded:: 3.1

Configure and run the Qt Installer Framework to generate a Qt installer.

.. only:: html

  .. contents::

Overview
^^^^^^^^

This :manual:`cpack generator <cpack-generators(7)>` generates
configuration and meta information for the `Qt Installer Framework
<https://doc.qt.io/qtinstallerframework/index.html>`_ (QtIFW),
and runs QtIFW tools to generate a Qt installer.

QtIFW provides tools and utilities to create installers for
the platforms supported by `Qt <https://www.qt.io>`_: Linux,
Microsoft Windows, and macOS.

To make use of this generator, QtIFW needs to be installed.
The :module:`CPackIFW` module looks for the location of the
QtIFW command-line utilities, and defines several commands to
control the behavior of this generator. See `Hints for Finding QtIFW`_.

Variables
^^^^^^^^^

You can use the following variables to change the behavior of the CPack ``IFW``
generator.

Debug
"""""

.. variable:: CPACK_IFW_VERBOSE

 .. versionadded:: 3.3

 Set to ``ON`` to enable addition debug output.
 By default is ``OFF``.

Package
"""""""

.. variable:: CPACK_IFW_PACKAGE_TITLE

 Name of the installer as displayed on the title bar.
 If not specified, it defaults to :variable:`CPACK_PACKAGE_DESCRIPTION_SUMMARY`.

.. variable:: CPACK_IFW_PACKAGE_PUBLISHER

 Publisher of the software (as shown in the Windows Control Panel).
 If not specified, it defaults to :variable:`CPACK_PACKAGE_VENDOR`.

.. variable:: CPACK_IFW_PRODUCT_URL

 URL to a page that contains product information on your web site.

.. variable:: CPACK_IFW_PACKAGE_ICON

 Filename for a custom installer icon. It must be an absolute path.
 This should be a ``.icns`` file on macOS and a ``.ico`` file on Windows.
 It is ignored on other platforms.

.. variable:: CPACK_IFW_PACKAGE_WINDOW_ICON

 Filename for a custom window icon in PNG format for the Installer
 application. It must be an absolute path.

.. variable:: CPACK_IFW_PACKAGE_LOGO

 Filename for a logo image in PNG format, used as ``QWizard::LogoPixmap``.
 It must be an absolute path.

.. variable:: CPACK_IFW_PACKAGE_WATERMARK

 .. versionadded:: 3.8

 Filename for a watermark image in PNG format, used as
 ``QWizard::WatermarkPixmap``. It must be an absolute path.

.. variable:: CPACK_IFW_PACKAGE_BANNER

 .. versionadded:: 3.8

 Filename for a banner image in PNG format, used as ``QWizard::BannerPixmap``.
 It must be an absolute path.

.. variable:: CPACK_IFW_PACKAGE_BACKGROUND

 .. versionadded:: 3.8

 Filename for a background image in PNG format, used as
 ``QWizard::BackgroundPixmap`` (only used by ``MacStyle``). It must be an
 absolute path.

.. variable:: CPACK_IFW_PACKAGE_WIZARD_STYLE

 .. versionadded:: 3.8

 Wizard style to be used (``Modern``, ``Mac``, ``Aero`` or ``Classic``).

.. variable:: CPACK_IFW_PACKAGE_WIZARD_DEFAULT_WIDTH

 .. versionadded:: 3.8

 Default width of the wizard in pixels. Setting a banner image will override
 this.

.. variable:: CPACK_IFW_PACKAGE_WIZARD_DEFAULT_HEIGHT

 .. versionadded:: 3.8

 Default height of the wizard in pixels. Setting a watermark image will
 override this.

.. variable:: CPACK_IFW_PACKAGE_WIZARD_SHOW_PAGE_LIST

 .. versionadded:: 3.20

 Set to ``OFF`` if the widget listing installer pages on the left side of the
 wizard should not be shown.

 It is ``ON`` by default, but will only have an effect if using QtIFW 4.0 or
 later.

.. variable:: CPACK_IFW_PACKAGE_TITLE_COLOR

 .. versionadded:: 3.8

 Color of the titles and subtitles (takes an HTML color code, such as
 ``#88FF33``).

.. variable:: CPACK_IFW_PACKAGE_STYLE_SHEET

 .. versionadded:: 3.15

 Filename for a stylesheet. It must be an absolute path.

.. variable:: CPACK_IFW_TARGET_DIRECTORY

 Default target directory for installation.
 If :variable:`CPACK_PACKAGE_INSTALL_DIRECTORY` is set, this defaults to
 ``@ApplicationsDir@/${CPACK_PACKAGE_INSTALL_DIRECTORY}``. If that variable
 isn't set either, the default used is ``@RootDir@/usr/local``.
 Predefined variables of the form ``@...@`` are expanded by the
 `QtIFW scripting engine <https://doc.qt.io/qtinstallerframework/scripting.html>`_.

.. variable:: CPACK_IFW_ADMIN_TARGET_DIRECTORY

 Default target directory for installation with administrator rights.

 You can use predefined variables.

.. variable:: CPACK_IFW_PACKAGE_REMOVE_TARGET_DIR

 .. versionadded:: 3.11

 Set to ``OFF`` if the target directory should not be deleted when uninstalling.

 Is ``ON`` by default

.. variable:: CPACK_IFW_PACKAGE_GROUP

 The group, which will be used to configure the root package.

.. variable:: CPACK_IFW_PACKAGE_NAME

 The root package name, which will be used if the configuration group is not
 specified.

.. variable:: CPACK_IFW_PACKAGE_START_MENU_DIRECTORY

 .. versionadded:: 3.3

 Name of the default program group for the product in the Windows Start menu.
 If not specified, it defaults to :variable:`CPACK_IFW_PACKAGE_NAME`.

.. variable:: CPACK_IFW_PACKAGE_MAINTENANCE_TOOL_NAME

 .. versionadded:: 3.3

 Filename of the generated maintenance tool.
 The platform-specific executable file extension will be appended.

 If not specified, QtIFW provides a default name (``maintenancetool``).

.. variable:: CPACK_IFW_PACKAGE_MAINTENANCE_TOOL_INI_FILE

 .. versionadded:: 3.3

 Filename for the configuration of the generated maintenance tool.

 If not specified, QtIFW uses a default file name (``maintenancetool.ini``).

.. variable:: CPACK_IFW_PACKAGE_ALLOW_NON_ASCII_CHARACTERS

 .. versionadded:: 3.3

 Set to ``ON`` if the installation path can contain non-ASCII characters.
 Only supported for QtIFW 2.0 and later. Older QtIFW versions will always
 allow non-ASCII characters.

.. variable:: CPACK_IFW_PACKAGE_ALLOW_SPACE_IN_PATH

 .. versionadded:: 3.3

 Set to ``OFF`` if the installation path cannot contain space characters.

 Is ``ON`` for QtIFW less 2.0 tools.

.. variable:: CPACK_IFW_PACKAGE_DISABLE_COMMAND_LINE_INTERFACE

 .. versionadded:: 3.23

 Set to ``ON`` if command line interface features should be disabled.
 It is ``OFF`` by default and will only have an effect if using QtIFW 4.0 or
 later.

.. variable:: CPACK_IFW_PACKAGE_CONTROL_SCRIPT

 .. versionadded:: 3.3

 Filename for a custom installer control script.

.. variable:: CPACK_IFW_PACKAGE_RESOURCES

 .. versionadded:: 3.7

 List of additional resources (``.qrc`` files) to include in the installer
 binary. They should be specified as absolute paths and no two resource files
 can have the same file name.

 You can use the :command:`cpack_ifw_add_package_resources` command to resolve
 relative paths.

.. variable:: CPACK_IFW_PACKAGE_FILE_EXTENSION

 .. versionadded:: 3.10

 The target binary extension.

 On Linux, the name of the target binary is automatically extended with
 ``.run``, if you do not specify the extension.

 On Windows, the target is created as an application with the extension
 ``.exe``, which is automatically added, if not supplied.

 On Mac, the target is created as an DMG disk image with the extension
 ``.dmg``, which is automatically added, if not supplied.

.. variable:: CPACK_IFW_REPOSITORIES_ALL

 The list of remote repositories.

 The default value of this variable is computed by CPack and contains
 all repositories added with :command:`cpack_ifw_add_repository`
 or updated with :command:`cpack_ifw_update_repository`.

.. variable:: CPACK_IFW_DOWNLOAD_ALL

 If this is ``ON``, all components will be downloaded. If not set, the
 behavior is determined by whether :command:`cpack_configure_downloads` has
 been called with the ``ALL`` option or not.

.. variable:: CPACK_IFW_PACKAGE_PRODUCT_IMAGES

 .. versionadded:: 3.23

 A list of images to be shown on the ``PerformInstallationPage``. These
 must be absolute paths and the images must be in PNG format.

 This feature is available for QtIFW 4.0.0 and later.

.. variable:: CPACK_IFW_PACKAGE_PRODUCT_IMAGE_URLS

 .. versionadded:: 3.31

 A list of URLs associated with the ProductImages.
 Only used if  ``CPACK_IFW_PACKAGE_PRODUCT_IMAGES`` is defined
 and it has the same size.

 This feature is available for QtIFW 4.0.0 and later.

.. variable:: CPACK_IFW_PACKAGE_RUN_PROGRAM

 .. versionadded:: 3.23

 Command executed after the installer is finished, if the user accepts the
 action. Provide the full path to the application, as found when installed.
 This typically means the path should begin with the QtIFW predefined variable
 ``@TargetDir@``.

 This feature is available for QtIFW 4.0.0 and later.

.. variable:: CPACK_IFW_PACKAGE_RUN_PROGRAM_ARGUMENTS

 .. versionadded:: 3.23

 List of arguments passed to the program specified in
 :variable:`CPACK_IFW_PACKAGE_RUN_PROGRAM`.

 This feature is available for QtIFW 4.0.0 and later.

.. variable:: CPACK_IFW_PACKAGE_RUN_PROGRAM_DESCRIPTION

 .. versionadded:: 3.23

 Text shown next to the check box for running the program after the
 installation. If :variable:`CPACK_IFW_PACKAGE_RUN_PROGRAM` is set but no
 description is provided, QtIFW will use a default message like
 ``Run <Name> now``.

 This feature is available for QtIFW 4.0.0 and later.

.. variable:: CPACK_IFW_PACKAGE_SIGNING_IDENTITY

 .. versionadded:: 3.23

 Allows specifying a code signing identity to be used for signing the generated
 app bundle. Only available on macOS, ignored on other platforms.

.. variable:: CPACK_IFW_ARCHIVE_FORMAT

 .. versionadded:: 3.23

 Set the format used when packaging new component data archives. If you omit
 this option, the ``7z`` format will be used as a default. Supported formats:

 * 7z
 * zip
 * tar.gz
 * tar.bz2
 * tar.xz

 .. note::

  If the Qt Installer Framework tools were built without libarchive support,
  only ``7z`` format is supported.

 This feature is available for QtIFW 4.2.0 and later.

.. variable:: CPACK_IFW_ARCHIVE_COMPRESSION

 .. versionadded:: 3.23

 Archive compression level. The allowable values are:

 * 0 (*No compression*)
 * 1 (*Fastest compression*)
 * 3 (*Fast compression*)
 * 5 (*Normal compression*)
 * 7 (*Maximum compression*)
 * 9 (*Ultra compression*)

 If this variable is not set, QtIFW will use a default compression level,
 which will typically be 5 (*Normal compression*).

 .. note::

  Some formats do not support all the possible values. For example ``zip``
  compression only supports values from 1 to 7.

 This feature is available for QtIFW 4.2.0 and later.

Components
""""""""""

.. variable:: CPACK_IFW_RESOLVE_DUPLICATE_NAMES

 Resolve duplicate names when installing components with groups.

.. variable:: CPACK_IFW_PACKAGES_DIRECTORIES

 Additional prepared packages directories that will be used to resolve
 dependent components.

.. variable:: CPACK_IFW_REPOSITORIES_DIRECTORIES

 .. versionadded:: 3.10

 Additional prepared repository directories that will be used to resolve and
 repack dependent components.

 This feature is available for QtIFW 3.1 and later.

QtIFW Tools
"""""""""""

.. variable:: CPACK_IFW_FRAMEWORK_VERSION

 .. versionadded:: 3.3

 The version of the QtIFW tools that will be used. This variable is set
 by the :module:`CPackIFW` module.

The following variables provide the locations of the QtIFW
command-line tools as discovered by the :module:`CPackIFW` module.
These variables are cached, and may be configured if needed.

.. variable:: CPACK_IFW_ARCHIVEGEN_EXECUTABLE

 .. versionadded:: 3.19

 The path to ``archivegen``.

.. variable:: CPACK_IFW_BINARYCREATOR_EXECUTABLE

 The path to ``binarycreator``.

.. variable:: CPACK_IFW_REPOGEN_EXECUTABLE

 The path to ``repogen``.

.. variable:: CPACK_IFW_INSTALLERBASE_EXECUTABLE

 The path to ``installerbase``.

.. variable:: CPACK_IFW_DEVTOOL_EXECUTABLE

 The path to ``devtool``.

Hints for Finding QtIFW
"""""""""""""""""""""""

Generally, the CPack ``IFW`` generator automatically finds QtIFW tools.
The following (in order of precedence) can also be set to augment the
locations normally searched by :command:`find_program`:

.. variable:: CPACK_IFW_ROOT

  .. versionadded:: 3.9

  CMake variable

.. envvar:: CPACK_IFW_ROOT

  .. versionadded:: 3.9

  Environment variable

.. variable:: QTIFWDIR

  CMake variable

.. envvar:: QTIFWDIR

  Environment variable

.. note::
  The specified path should not contain ``bin`` at the end
  (for example: ``D:\\DevTools\\QtIFW2.0.5``).

Other Settings
^^^^^^^^^^^^^^

Online installer
""""""""""""""""

By default, this generator generates an *offline installer*. This means
that all packaged files are fully contained in the installer executable.

In contrast, an *online installer* will download some or all components from
a remote server.

The ``DOWNLOADED`` option in the :command:`cpack_add_component` command
specifies that a component is to be downloaded. Alternatively, the ``ALL``
option in the :command:`cpack_configure_downloads` command specifies that
``all`` components are to be be downloaded.

The :command:`cpack_ifw_add_repository` command and the
:variable:`CPACK_IFW_DOWNLOAD_ALL` variable allow for more specific
configuration.

When there are online components, CPack will write them to archive files.
The help page of the :module:`CPackComponent` module, especially the section
on the :command:`cpack_configure_downloads` function, explains how to make
these files accessible from a download URL.

Internationalization
""""""""""""""""""""

.. versionadded:: 3.9

Some variables and command arguments support internationalization via
CMake script. This is an optional feature.

Installers created by QtIFW tools have built-in support for
internationalization and many phrases are localized to many languages,
but this does not apply to the description of your components and groups.

Localization of the description of your components and groups is useful for
users of your installers.

A localized variable or argument can contain a single default value, and
after that a set of pairs with the name of the locale and the localized value.

For example:

.. code-block:: cmake

   set(LOCALIZABLE_VARIABLE "Default value"
     en "English value"
     en_US "American value"
     en_GB "Great Britain value"
     )

See Also
^^^^^^^^

Qt Installer Framework Manual:

* Index page:
  https://doc.qt.io/qtinstallerframework/index.html

* Component Scripting:
  https://doc.qt.io/qtinstallerframework/scripting.html

* Predefined Variables:
  https://doc.qt.io/qtinstallerframework/scripting.html#predefined-variables

* Promoting Updates:
  https://doc.qt.io/qtinstallerframework/ifw-updates.html

Download Qt Installer Framework for your platform from Qt site:
 https://download.qt.io/official_releases/qt-installer-framework
