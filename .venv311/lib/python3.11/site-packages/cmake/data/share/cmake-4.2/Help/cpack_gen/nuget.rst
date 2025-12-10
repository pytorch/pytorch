CPack NuGet Generator
---------------------

.. versionadded:: 3.12

When build a NuGet package there is no direct way to control an output
filename due a lack of the corresponding CLI option of NuGet, so there
is no :variable:`!CPACK_NUGET_PACKAGE_FILE_NAME` variable. To form the output filename
NuGet uses the package name and the version according to its built-in rules.

Also, be aware that including a top level directory
(:variable:`CPACK_INCLUDE_TOPLEVEL_DIRECTORY`) is ignored by this generator.


Variables specific to CPack NuGet generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CPack NuGet generator may be used to create NuGet packages using
:module:`CPack`. The CPack NuGet generator is a :module:`CPack` generator thus
it uses the :variable:`!CPACK_XXX` variables used by :module:`CPack`.

The CPack NuGet generator has specific features which are controlled by the
specifics :variable:`!CPACK_NUGET_XXX` variables. In the "one per group" mode
(see :variable:`CPACK_COMPONENTS_GROUPING`), ``<compName>`` placeholder
in the variables below would contain a group name (uppercased and turned into
a "C" identifier).

List of CPack NuGet generator specific variables:

.. variable:: CPACK_NUGET_COMPONENT_INSTALL

 Enable component packaging for CPack NuGet generator

 :Mandatory: No
 :Default: ``OFF``

.. variable:: CPACK_NUGET_PACKAGE_DEBUG

 Enable debug messages while executing CPack NuGet generator.

 :Mandatory: No
 :Default: ``OFF``

.. variable:: CPACK_NUGET_SYMBOL_PACKAGE

 .. versionadded:: 4.1

 Generate a Nuget symbol package (.snupkg).

 :Mandatory: No
 :Default: ``OFF``

Required metadata variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. variable:: CPACK_NUGET_PACKAGE_NAME
              CPACK_NUGET_<compName>_PACKAGE_NAME

 The NUGET package name. ``CPACK_NUGET_PACKAGE_NAME`` is used as the
 package ``id`` on nuget.org_

 :Mandatory: Yes
 :Default: :variable:`CPACK_PACKAGE_NAME`

.. variable:: CPACK_NUGET_PACKAGE_VERSION
              CPACK_NUGET_<compName>_PACKAGE_VERSION

 The NuGet package version.

 :Mandatory: Yes
 :Default: :variable:`CPACK_PACKAGE_VERSION`

.. variable:: CPACK_NUGET_PACKAGE_DESCRIPTION
              CPACK_NUGET_<compName>_PACKAGE_DESCRIPTION

 A long description of the package for UI display.

 :Mandatory: Yes
 :Default:

    - :variable:`CPACK_COMPONENT_<compName>_DESCRIPTION`,
    - :variable:`!CPACK_COMPONENT_GROUP_<groupName>_DESCRIPTION`,
    - :variable:`CPACK_PACKAGE_DESCRIPTION`

.. variable:: CPACK_NUGET_PACKAGE_AUTHORS
              CPACK_NUGET_<compName>_PACKAGE_AUTHORS

 A comma-separated list of packages authors, matching the profile names
 on nuget.org_. These are displayed in the NuGet Gallery on
 nuget.org_ and are used to cross-reference packages by the same
 authors.

 :Mandatory: Yes
 :Default: :variable:`CPACK_PACKAGE_VENDOR`


Optional metadata variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. variable:: CPACK_NUGET_PACKAGE_OWNERS
              CPACK_NUGET_<compName>_PACKAGE_OWNERS

 .. deprecated:: 3.30
  Use authors (:variable:`CPACK_NUGET_PACKAGE_AUTHORS`) instead.

 A comma-separated list of the package creators using profile names
 on nuget.org_. This is often the same list as in authors,
 and is ignored when uploading the package to nuget.org_.

 :Mandatory: No
 :Default: None

.. variable:: CPACK_NUGET_PACKAGE_HOMEPAGE_URL
              CPACK_NUGET_<compName>_PACKAGE_HOMEPAGE_URL

 An URL for the package's home page, often shown in UI displays as well
 as nuget.org_.

 :Mandatory: No
 :Default: :variable:`CPACK_PACKAGE_HOMEPAGE_URL`

.. variable:: CPACK_NUGET_PACKAGE_LICENSEURL
              CPACK_NUGET_<compName>_PACKAGE_LICENSEURL

 .. deprecated:: 3.20
  Use a local license file
  (:variable:`CPACK_NUGET_PACKAGE_LICENSE_FILE_NAME`)
  or a `SPDX license identifier`_
  (:variable:`CPACK_NUGET_PACKAGE_LICENSE_EXPRESSION`) instead.

 An URL for the package's license, often shown in UI displays as well
 as on nuget.org_.

 :Mandatory: No
 :Default: None
 :Supported: NuGet 4.9.0 and above

.. variable:: CPACK_NUGET_PACKAGE_LICENSE_FILE_NAME
              CPACK_NUGET_<compName>_PACKAGE_LICENSE_FILE_NAME

 .. versionadded:: 3.20

 The package's license file in :file:`.txt` or :file:`.md` format.

 If :variable:`!CPACK_NUGET_PACKAGE_LICENSE_FILE_NAME` is specified,
 :variable:`!CPACK_NUGET_PACKAGE_LICENSE_EXPRESSION` is ignored.

 :Mandatory: No
 :Default: None
 :Supported: NuGet 4.9.0 and above

.. variable:: CPACK_NUGET_PACKAGE_LICENSE_EXPRESSION
              CPACK_NUGET_<compName>_PACKAGE_LICENSE_EXPRESSION

 .. versionadded:: 3.20

 A Software Package Data Exchange `SPDX license identifier`_ such as
 ``MIT``, ``BSD-3-Clause``, or ``LGPL-3.0-or-later``. In the case of a
 choice of licenses or more complex restrictions, compound license
 expressions may be formed using boolean operators, for example
 ``MIT OR BSD-3-Clause``.  See the `SPDX specification`_ for guidance
 on forming complex license expressions.

 If :variable:`CPACK_NUGET_PACKAGE_LICENSE_FILE_NAME` is specified,
 :variable:`!CPACK_NUGET_PACKAGE_LICENSE_EXPRESSION` is ignored.

 :Mandatory: No
 :Default: None

.. variable:: CPACK_NUGET_PACKAGE_ICONURL
              CPACK_NUGET_<compName>_PACKAGE_ICONURL

 .. deprecated:: 3.20
  Use a local icon file (:variable:`CPACK_NUGET_PACKAGE_ICON`) instead.

 An URL for a 64x64 image with transparency background to use as the
 icon for the package in UI display.

 :Mandatory: No
 :Default: None

.. variable:: CPACK_NUGET_PACKAGE_ICON
              CPACK_NUGET_<compName>_PACKAGE_ICON

 .. versionadded:: 3.20

 The filename of a 64x64 image with transparency background to use as the
 icon for the package in UI display.

 :Mandatory: No
 :Default: None
 :Supported: NuGet 5.3.0 and above


.. variable:: CPACK_NUGET_PACKAGE_README
              CPACK_NUGET_<compName>_PACKAGE_README

 .. versionadded:: 3.30

 The package path relative to the root of the package to a readme file.
 Supported file formats include only Markdown (``*.md``).

 :Mandatory: No
 :Default: None
 :Supported: NuGet 5.10.0 preview 2 and above

.. variable:: CPACK_NUGET_PACKAGE_REQUIRE_LICENSE_ACCEPTANCE

 When set to a true value, the user will be prompted to accept the license
 before installing the package.

 :Mandatory: No
 :Default: None

.. variable:: CPACK_NUGET_PACKAGE_DESCRIPTION_SUMMARY
              CPACK_NUGET_<compName>_PACKAGE_DESCRIPTION_SUMMARY

 .. deprecated:: 3.30
  Summary is being deprecated. Use description
  (:variable:`CPACK_NUGET_PACKAGE_DESCRIPTION`) instead.

 A short description of the package for UI display. If omitted, a
 truncated version of description is used.

 :Mandatory: No
 :Default: :variable:`CPACK_PACKAGE_DESCRIPTION_SUMMARY`

.. variable:: CPACK_NUGET_PACKAGE_RELEASE_NOTES
              CPACK_NUGET_<compName>_PACKAGE_RELEASE_NOTES

 A description of the changes made in this release of the package,
 often used in UI like the Updates tab of the Visual Studio Package
 Manager in place of the package description.

 :Mandatory: No
 :Default: None

.. variable:: CPACK_NUGET_PACKAGE_COPYRIGHT
              CPACK_NUGET_<compName>_PACKAGE_COPYRIGHT

 Copyright details for the package.

 :Mandatory: No
 :Default: None

.. variable:: CPACK_NUGET_PACKAGE_LANGUAGE
              CPACK_NUGET_<compName>_PACKAGE_LANGUAGE

 .. versionadded:: 3.20

 Locale specifier for the package, for example ``en_CA``.

 :Mandatory: No
 :Default: None

.. variable:: CPACK_NUGET_PACKAGE_TAGS
              CPACK_NUGET_<compName>_PACKAGE_TAGS

 A space-delimited list of tags and keywords that describe the
 package and aid discoverability of packages through search and
 filtering.

 :Mandatory: No
 :Default: None

.. variable:: CPACK_NUGET_PACKAGE_REPOSITORY_URL
              CPACK_NUGET_<compName>_REPOSITORY_URL

  .. versionadded:: 3.30

 Repository metadata allows you to map the ``*.nupkg`` to the repository
 that built it. This should be a publicly available URL that can be invoked
 directly by a version control software. It should not be an HTML page as
 this is meant for the computer.

 :Mandatory: No
 :Default: None
 :Supported: NuGet 4.0 and above

.. variable:: CPACK_NUGET_PACKAGE_REPOSITORY_TYPE
              CPACK_NUGET_<compName>_REPOSITORY_TYPE

  .. versionadded:: 3.30

 A type of the VCS repository.  When uploading a package to nuget.org_, the
 type is limited to 100 characters.

 :Mandatory: Yes, if repository URL has been specified
 :Default: None
 :Supported: NuGet 4.0 and above

.. variable:: CPACK_NUGET_PACKAGE_REPOSITORY_BRANCH
              CPACK_NUGET_<compName>_REPOSITORY_BRANCH

  .. versionadded:: 3.30

 A VSC branch name to build the package.

 :Mandatory: No
 :Default: None
 :Supported: NuGet 4.6 and above

.. variable:: CPACK_NUGET_PACKAGE_REPOSITORY_COMMIT
              CPACK_NUGET_<compName>_REPOSITORY_COMMIT

  .. versionadded:: 3.30

 A SHA-1 hash of the commit to build the package.

 :Mandatory: No
 :Default: None
 :Supported: NuGet 4.6 and above

.. variable:: CPACK_NUGET_PACKAGE_TITLE
              CPACK_NUGET_<compName>_PACKAGE_TITLE

 A human-friendly title of the package, typically used in UI displays
 as on nuget.org_ and the Package Manager in Visual Studio. If not
 specified, the package ID is used.

 :Mandatory: No
 :Default:

    - :variable:`CPACK_COMPONENT_<compName>_DISPLAY_NAME`,
    - :variable:`!CPACK_COMPONENT_GROUP_<groupName>_DISPLAY_NAME`


Dependencies specification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. variable:: CPACK_NUGET_PACKAGE_DEPENDENCIES
              CPACK_NUGET_<compName>_PACKAGE_DEPENDENCIES

 A list of default (not framework-specific) package dependencies.

 :Mandatory: No
 :Default: None

.. variable:: CPACK_NUGET_PACKAGE_DEPENDENCIES_<dependency>_VERSION
              CPACK_NUGET_<compName>_PACKAGE_DEPENDENCIES_<dependency>_VERSION

 A `version specification`_ for the particular dependency, where
 ``<dependency>`` is an item of the dependency list (see above).

 :Mandatory: No
 :Default: None

.. variable:: CPACK_NUGET_PACKAGE_TFMS
              CPACK_NUGET_<compName>_PACKAGE_TFMS

 .. versionadded:: 3.30

 A list of Target Framework Monikers (TFMs) for the package, e.g., "net47;netcoreapp21".
 For each of these TFMs a `dependency group`_ will be generated in the dependencies block of the NuGet
 package. Framework-specific dependencies can be added to these groups with the TFM
 dependency lists (see below).

 This variable is particularly useful for fixing warnings `NU5128`_.

 :Mandatory: No
 :Default: None

.. variable:: CPACK_NUGET_PACKAGE_DEPENDENCIES_<tfm>
              CPACK_NUGET_<compName>_PACKAGE_DEPENDENCIES_<tfm>

 .. versionadded:: 3.30

 A list of package dependencies that apply specifically to the ``<tfm>`` framework, where ``<tfm>``
 is an item from the TFMs list (see above).

 :Mandatory: No
 :Default: None

.. variable:: CPACK_NUGET_PACKAGE_DEPENDENCIES_<tfm>_<dependency>_VERSION
              CPACK_NUGET_<compName>_PACKAGE_DEPENDENCIES_<tfm>_<dependency>_VERSION

 .. versionadded:: 3.30

 A `version specification`_ for the particular framework-specific dependency, where
 ``<dependency>`` is an item of the ``<tfm>``-specific dependency list (see above).

 :Mandatory: No
 :Default: None


Example usage
^^^^^^^^^^^^^

.. code-block:: cmake

  set(CPACK_GENERATOR NuGet)
  # Set up package metadata
  set(CPACK_PACKAGE_NAME SamplePackage)
  set(CPACK_PACKAGE_VERSION "1.0.0")
  set(CPACK_PACKAGE_VENDOR "Example Inc")
  set(CPACK_NUGET_PACKAGE_AUTHORS "ExampleInc")
  set(CPACK_PACKAGE_DESCRIPTION "A .NET wrapper around the foobar library for frobbling bratchens")
  set(CPACK_PACKAGE_HOMEPAGE_URL "https://www.example.com")
  set(CPACK_NUGET_PACKAGE_REPOSITORY_URL "https://github.com/example/libfoobar.git")
  set(CPACK_NUGET_PACKAGE_REPOSITORY_TYPE git)
  set(CPACK_NUGET_PACKAGE_LICENSE_EXPRESSION "MIT")
  # Set up dependencies
  set(CPACK_NUGET_PACKAGE_TFMS "net4;net6.0")
  set(CPACK_NUGET_PACKAGE_DEPENDENCIES_net4 "Foo;Bar")
  # NB: If a version number is omitted, the dependency will not be created
  set(CPACK_NUGET_PACKAGE_DEPENDENCIES_net4_Foo_VERSION "1.23")
  set(CPACK_NUGET_PACKAGE_DEPENDENCIES_net4_Bar_VERSION "4.3.2")
  # NB: General dependencies (not framework-specific) go in this variable
  set(CPACK_NUGET_PACKAGE_DEPENDENCIES "Baz")
  set(CPACK_NUGET_PACKAGE_DEPENDENCIES_Baz_VERSION "9.8.6")
  # NB: Since "net6.0" was listed but no dependencies have been specified, an empty group
  # will be added to the nuspec file for this framework. This can be used to address warning NU5128.

  include(CPack)


.. _nuget.org: https://www.nuget.org
.. _version specification: https://learn.microsoft.com/en-us/nuget/concepts/package-versioning#version-ranges
.. _SPDX license identifier: https://spdx.org/licenses
.. _SPDX specification: https://spdx.github.io/spdx-spec/v2.3/SPDX-license-expressions
.. _dependency group: https://learn.microsoft.com/en-us/nuget/reference/nuspec#dependency-groups
.. _NU5128: https://learn.microsoft.com/en-us/nuget/reference/errors-and-warnings/nu5128

.. NuGet spec docs https://docs.microsoft.com/en-us/nuget/reference/nuspec
