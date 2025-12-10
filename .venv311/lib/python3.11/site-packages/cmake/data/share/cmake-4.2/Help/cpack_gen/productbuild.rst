CPack productbuild Generator
----------------------------

.. versionadded:: 3.7

productbuild CPack generator (macOS).

Variables specific to CPack productbuild generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following variable is specific to installers built on Mac
macOS using ProductBuild:

.. variable:: CPACK_COMMAND_PRODUCTBUILD

 Path to the ``productbuild(1)`` command used to generate a product archive for
 the macOS Installer or Mac App Store.  This variable can be used to override
 the automatically detected command (or specify its location if the
 auto-detection fails to find it).

.. variable:: CPACK_PRODUCTBUILD_IDENTIFIER

 .. versionadded:: 3.23

 Set the unique (non-localized) product identifier to be associated with the
 product (i.e., ``com.kitware.cmake``). Any component product names will be
 appended to this value.

.. variable:: CPACK_PRODUCTBUILD_IDENTITY_NAME

 .. versionadded:: 3.8

 Adds a digital signature to the resulting package.


.. variable:: CPACK_PRODUCTBUILD_KEYCHAIN_PATH

 .. versionadded:: 3.8

 Specify a specific keychain to search for the signing identity.


.. variable:: CPACK_COMMAND_PKGBUILD

 Path to the ``pkgbuild(1)`` command used to generate an macOS component package
 on macOS.  This variable can be used to override the automatically detected
 command (or specify its location if the auto-detection fails to find it).


.. variable:: CPACK_PKGBUILD_IDENTITY_NAME

 .. versionadded:: 3.8

 Adds a digital signature to the resulting package.


.. variable:: CPACK_PKGBUILD_KEYCHAIN_PATH

 .. versionadded:: 3.8

 Specify a specific keychain to search for the signing identity.


.. variable:: CPACK_PREFLIGHT_<COMP>_SCRIPT

 Full path to a file that will be used as the ``preinstall`` script for the
 named ``<COMP>`` component's package, where ``<COMP>`` is the uppercased
 component name.  No ``preinstall`` script is added if this variable is not
 defined for a given component.


.. variable:: CPACK_POSTFLIGHT_<COMP>_SCRIPT

 Full path to a file that will be used as the ``postinstall`` script for the
 named ``<COMP>`` component's package, where ``<COMP>`` is the uppercased
 component name.  No ``postinstall`` script is added if this variable is not
 defined for a given component.

.. variable:: CPACK_PRODUCTBUILD_RESOURCES_DIR

 .. versionadded:: 3.9

 If specified the productbuild generator copies files from this directory
 (including subdirectories) to the ``Resources`` directory. This is done
 before the :variable:`CPACK_RESOURCE_FILE_WELCOME`,
 :variable:`CPACK_RESOURCE_FILE_README`, and
 :variable:`CPACK_RESOURCE_FILE_LICENSE` files are copied.

.. variable:: CPACK_PRODUCTBUILD_DOMAINS

 .. versionadded:: 3.23

 This option enables more granular control over where the product may be
 installed. When it is set to true (see policy :policy:`CMP0161`), a
 ``domains`` element of the following form will be added to the
 productbuild Distribution XML:

 .. code-block:: xml

    <domains enable_anywhere="true" enable_currentUserHome="false" enable_localSystem="true"/>

 The default values are as shown above, but can be overridden with
 :variable:`CPACK_PRODUCTBUILD_DOMAINS_ANYWHERE`,
 :variable:`CPACK_PRODUCTBUILD_DOMAINS_USER`, and
 :variable:`CPACK_PRODUCTBUILD_DOMAINS_ROOT`.

.. variable:: CPACK_PRODUCTBUILD_DOMAINS_ANYWHERE

 .. versionadded:: 3.23

 May be used to override the ``enable_anywhere`` attribute in the ``domains``
 element of the Distribution XML. When set to true, the product can be
 installed at the root of any volume, including non-system volumes.

 :variable:`CPACK_PRODUCTBUILD_DOMAINS` must be set to true for this variable
 to have any effect.

.. variable:: CPACK_PRODUCTBUILD_DOMAINS_USER

 .. versionadded:: 3.23

 May be used to override the ``enable_currentUserHome`` attribute in the
 ``domains`` element of the Distribution XML. When set to true, the product
 can be installed into the current user's home directory. Note that when
 installing into the user's home directory, the following additional
 requirements will apply:

 * The installer may not write outside the user's home directory.
 * The install will be performed as the current user rather than as ``root``.
   This may have ramifications for :variable:`CPACK_PREFLIGHT_<COMP>_SCRIPT`
   and :variable:`CPACK_POSTFLIGHT_<COMP>_SCRIPT`.
 * Administrative privileges will not be needed to perform the install.

 :variable:`CPACK_PRODUCTBUILD_DOMAINS` must be set to true for this variable
 to have any effect.

.. variable:: CPACK_PRODUCTBUILD_DOMAINS_ROOT

 .. versionadded:: 3.23

 May be used to override the ``enable_localSystem`` attribute in the
 ``domains`` element of the Distribution XML. When set to true, the product
 can be installed in the root directory. This should normally be set to true
 unless the product should only be installed to the user's home directory.

 :variable:`CPACK_PRODUCTBUILD_DOMAINS` must be set to true for this variable
 to have any effect.

Background Image
""""""""""""""""

.. versionadded:: 3.17

This group of variables controls the background image of the generated
installer.

.. variable:: CPACK_PRODUCTBUILD_BACKGROUND

 Adds a background to Distribution XML if specified. The value contains the
 path to image in ``Resources`` directory.

.. variable:: CPACK_PRODUCTBUILD_BACKGROUND_ALIGNMENT

 Adds an ``alignment`` attribute to the background in Distribution XML.
 Refer to Apple documentation for valid values.

.. variable:: CPACK_PRODUCTBUILD_BACKGROUND_SCALING

 Adds a ``scaling`` attribute to the background in Distribution XML.
 Refer to Apple documentation for valid values.

.. variable:: CPACK_PRODUCTBUILD_BACKGROUND_MIME_TYPE

 Adds a ``mime-type`` attribute to the background in Distribution XML.
 The option contains MIME type of an image.

.. variable:: CPACK_PRODUCTBUILD_BACKGROUND_UTI

 Adds an ``uti`` attribute to the background in Distribution XML.
 The option contains UTI type of an image.

.. variable:: CPACK_PRODUCTBUILD_BACKGROUND_DARKAQUA

 Adds a background for the Dark Aqua theme to Distribution XML if
 specified. The value contains the path to image in ``Resources``
 directory.

.. variable:: CPACK_PRODUCTBUILD_BACKGROUND_DARKAQUA_ALIGNMENT

 Does the same as :variable:`CPACK_PRODUCTBUILD_BACKGROUND_ALIGNMENT` option,
 but for the dark theme.

.. variable:: CPACK_PRODUCTBUILD_BACKGROUND_DARKAQUA_SCALING

 Does the same as :variable:`CPACK_PRODUCTBUILD_BACKGROUND_SCALING` option,
 but for the dark theme.

.. variable:: CPACK_PRODUCTBUILD_BACKGROUND_DARKAQUA_MIME_TYPE

 Does the same as :variable:`CPACK_PRODUCTBUILD_BACKGROUND_MIME_TYPE` option,
 but for the dark theme.

.. variable:: CPACK_PRODUCTBUILD_BACKGROUND_DARKAQUA_UTI

 Does the same as :variable:`CPACK_PRODUCTBUILD_BACKGROUND_UTI` option,
 but for the dark theme.

Distribution XML Template
^^^^^^^^^^^^^^^^^^^^^^^^^

CPack uses a template file to generate the ``distribution.dist`` file used
internally by this package generator. Ordinarily, CMake provides the template
file, but projects may supply their own by placing a file called
``CPack.distribution.dist.in`` in one of the directories listed in the
:variable:`CMAKE_MODULE_PATH` variable. CPack will then pick up the project's
template file instead of using its own.

The ``distribution.dist`` file is generated by performing substitutions
similar to the :command:`configure_file` command. Any variable set when
CPack runs will be available for substitution using the usual ``@...@``
form. The following variables are also set internally and made available for
substitution:

``CPACK_RESOURCE_FILE_LICENSE_NOPATH``
  Same as :variable:`CPACK_RESOURCE_FILE_LICENSE` except without the path.
  The named file will be available in the same directory as the generated
  ``distribution.dist`` file.

``CPACK_RESOURCE_FILE_README_NOPATH``
  Same as :variable:`CPACK_RESOURCE_FILE_README` except without the path.
  The named file will be available in the same directory as the generated
  ``distribution.dist`` file.

``CPACK_RESOURCE_FILE_WELCOME_NOPATH``
  Same as :variable:`CPACK_RESOURCE_FILE_WELCOME` except without the path.
  The named file will be available in the same directory as the generated
  ``distribution.dist`` file.

``CPACK_APPLE_PKG_INSTALLER_CONTENT``
  .. versionadded:: 3.23

  This contains all the XML elements that specify installer-wide options
  (including domain details), default backgrounds and the choices outline.

``CPACK_PACKAGEMAKER_CHOICES``
  .. deprecated:: 3.23

  This contains only the XML elements that specify the default backgrounds
  and the choices outline. It does not include the installer-wide options or
  any domain details. Use ``CPACK_APPLE_PKG_INSTALLER_CONTENT`` instead.
