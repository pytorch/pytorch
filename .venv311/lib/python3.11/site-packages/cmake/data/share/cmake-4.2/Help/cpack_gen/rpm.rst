CPack RPM Generator
-------------------

The built in (binary) CPack RPM generator (Unix only)

Variables specific to CPack RPM generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CPack RPM generator may be used to create RPM packages using :module:`CPack`.
The CPack RPM generator is a :module:`CPack` generator thus it uses the
:variable:`!CPACK_XXX` variables used by :module:`CPack`.

The CPack RPM generator has specific features which are controlled by the specifics
:variable:`!CPACK_RPM_XXX` variables.

:variable:`!CPACK_RPM_<COMPONENT>_XXXX` variables may be used in order to have
**component-specific** values.  Note however that ``<COMPONENT>`` refers to the
**grouping name** written in upper case. It may be either a component name or
a component GROUP name. Usually, those variables correspond to RPM spec file
entities. One may find information about spec files here
https://rpm.org/documentation.

.. versionchanged:: 3.6

 ``<COMPONENT>`` part of variables is preferred to be in upper case (e.g. if
 component is named ``foo`` then use :variable:`!CPACK_RPM_FOO_XXXX` variable
 name format) as is with other :variable:`!CPACK_<COMPONENT>_XXXX` variables.
 For the purposes of back compatibility (CMake/CPack version 3.5 and lower)
 support for same cased component (e.g. ``fOo`` would be used as
 :variable:`!CPACK_RPM_fOo_XXXX`) is still supported for variables defined in
 older versions of CMake/CPack but is not guaranteed for variables that
 will be added in the future. For the sake of back compatibility same cased
 component variables also override upper cased versions where both are
 present.

Here are some CPack RPM generator wiki resources that are here for historic
reasons and are no longer maintained but may still prove useful:

- https://gitlab.kitware.com/cmake/community/-/wikis/doc/cpack/Configuration
- https://gitlab.kitware.com/cmake/community/-/wikis/doc/cpack/PackageGenerators#rpm-unix-only

List of CPack RPM generator specific variables:

.. variable:: CPACK_RPM_COMPONENT_INSTALL

 Enable component packaging for CPack RPM generator

 :Mandatory: No
 :Default: ``OFF``

 If enabled (``ON``) multiple packages are generated. By default
 a single package containing files of all components is generated.

.. variable:: CPACK_RPM_PACKAGE_SUMMARY
              CPACK_RPM_<component>_PACKAGE_SUMMARY

 The RPM package summary.

 :Mandatory: Yes
 :Default: :variable:`CPACK_PACKAGE_DESCRIPTION_SUMMARY`

 .. versionadded:: 3.2
  Per-component :variable:`!CPACK_RPM_<component>_PACKAGE_SUMMARY` variables.

.. variable:: CPACK_RPM_PACKAGE_NAME
              CPACK_RPM_<component>_PACKAGE_NAME

 The RPM package name.

 :Mandatory: Yes
 :Default: :variable:`CPACK_PACKAGE_NAME`

 .. versionadded:: 3.5
  Per-component :variable:`!CPACK_RPM_<component>_PACKAGE_NAME` variables.

.. variable:: CPACK_RPM_FILE_NAME
              CPACK_RPM_<component>_FILE_NAME

 .. versionadded:: 3.6

 Package file name.

 :Mandatory: Yes
 :Default: ``<CPACK_PACKAGE_FILE_NAME>[-<component>].rpm`` with spaces
               replaced by '-'

 This may be set to:

 ``RPM-DEFAULT``
    Tell ``rpmbuild`` to automatically generate the package file name.

 ``<file-name>[.rpm]``
   Use the given file name.

   .. versionchanged:: 3.29

     The ``.rpm`` suffix will be automatically added if missing.
     Previously the suffix was required.

 .. note::

   By using user provided spec file, rpm macro extensions such as for
   generating ``debuginfo`` packages or by simply using multiple components more
   than one rpm file may be generated, either from a single spec file or from
   multiple spec files (each component execution produces its own spec file).
   In such cases duplicate file names may occur as a result of this variable
   setting or spec file content structure. Duplicate files get overwritten
   and it is up to the packager to set the variables in a manner that will
   prevent such errors.

.. variable:: CPACK_RPM_MAIN_COMPONENT

 .. versionadded:: 3.8

 Main component that is packaged without component suffix.

 :Mandatory: No
 :Default:

 This variable can be set to any component or group name so that component or
 group rpm package is generated without component suffix in filename and
 package name.

.. variable:: CPACK_RPM_PACKAGE_EPOCH

 .. versionadded:: 3.10

 The RPM package epoch

 :Mandatory: No
 :Default:

 Optional number that should be incremented when changing versioning schemas
 or fixing mistakes in the version numbers of older packages.

.. variable:: CPACK_RPM_PACKAGE_VERSION

 The RPM package version.

 :Mandatory: Yes
 :Default: :variable:`CPACK_PACKAGE_VERSION`

.. variable:: CPACK_RPM_PACKAGE_ARCHITECTURE
              CPACK_RPM_<component>_PACKAGE_ARCHITECTURE

 The RPM package architecture.

 :Mandatory: Yes
 :Default: Native architecture output by ``uname -m``

 This may be set to ``noarch`` if you know you are building a ``noarch`` package.

 .. versionadded:: 3.3
  Per-component :variable:`!CPACK_RPM_<component>_PACKAGE_ARCHITECTURE` variables.

.. variable:: CPACK_RPM_PACKAGE_RELEASE

 The RPM package release.

 :Mandatory: Yes
 :Default: 1

 This is the numbering of the RPM package itself, i.e. the version of the
 packaging and not the version of the content (see
 :variable:`CPACK_RPM_PACKAGE_VERSION`). One may change the default value if
 the previous packaging was buggy and/or you want to put here a fancy Linux
 distro specific numbering.

.. note::

 This is the string that goes into the RPM ``Release:`` field. Some distros
 (e.g. Fedora, CentOS) require ``1%{?dist}`` format and not just a number.
 ``%{?dist}`` part can be added by setting :variable:`CPACK_RPM_PACKAGE_RELEASE_DIST`.

.. variable:: CPACK_RPM_PACKAGE_RELEASE_DIST

 .. versionadded:: 3.6

 The dist tag that is added  RPM ``Release:`` field.

 :Mandatory: No
 :Default: ``OFF``

 This is the reported ``%{dist}`` tag from the current distribution or empty
 ``%{dist}`` if RPM macro is not set. If this variable is set then RPM
 ``Release:`` field value is set to ``${CPACK_RPM_PACKAGE_RELEASE}%{?dist}``.

.. variable:: CPACK_RPM_PACKAGE_LICENSE

 The RPM package license policy.

 :Mandatory: Yes
 :Default: "unknown"

.. variable:: CPACK_RPM_PACKAGE_GROUP
              CPACK_RPM_<component>_PACKAGE_GROUP

 The RPM package group.

 :Mandatory: Yes
 :Default: "unknown"

 .. versionadded:: 3.5
  Per-component :variable:`!CPACK_RPM_<component>_PACKAGE_GROUP` variables.

.. variable:: CPACK_RPM_PACKAGE_VENDOR

 The RPM package vendor.

 :Mandatory: Yes
 :Default: CPACK_PACKAGE_VENDOR if set or "unknown"

.. variable:: CPACK_RPM_PACKAGE_URL
              CPACK_RPM_<component>_PACKAGE_URL

 The projects URL.

 :Mandatory: No
 :Default: :variable:`CMAKE_PROJECT_HOMEPAGE_URL`

 .. versionadded:: 3.12
  The :variable:`!CMAKE_PROJECT_HOMEPAGE_URL` variable.

.. variable:: CPACK_RPM_PACKAGE_DESCRIPTION
              CPACK_RPM_<component>_PACKAGE_DESCRIPTION

 RPM package description.

 :Mandatory: Yes
 :Default:

   - :variable:`CPACK_COMPONENT_<compName>_DESCRIPTION`
     (component based installers only) if set,
   - :variable:`CPACK_PACKAGE_DESCRIPTION_FILE`
     if set, or
   - ``no package description available``

 .. versionadded:: 3.2
  Per-component :variable:`!CPACK_RPM_<component>_PACKAGE_DESCRIPTION` variables.

.. variable:: CPACK_RPM_COMPRESSION_TYPE

 RPM compression type.

 :Mandatory: No
 :Default: (system default)

 May be used to override RPM compression type to be used to build the
 RPM. For example some Linux distributions default to ``xz`` or ``zstd``.
 Using this, one can specify a specific compression type to be used.

 Possible values are:

  ``lzma``
    Lempel–Ziv–Markov chain algorithm

  ``xz``
    XZ Utils compression

  ``bzip2``
    bzip2 Burrows–Wheeler algorithm

  ``gzip``
    GNU Gzip compression

  ``zstd``
    .. versionadded:: 3.31

    Zstandard compression

.. variable:: CPACK_RPM_PACKAGE_AUTOREQ
              CPACK_RPM_<component>_PACKAGE_AUTOREQ

 RPM spec autoreq field.

 :Mandatory: No
 :Default:

 May be used to enable (``1``, ``yes``) or disable (``0``, ``no``) automatic
 shared libraries dependency detection. Dependencies are added to requires list.

 .. note::

   By default automatic dependency detection is enabled by rpm generator.

.. variable:: CPACK_RPM_PACKAGE_AUTOPROV
              CPACK_RPM_<component>_PACKAGE_AUTOPROV

 RPM spec autoprov field.

 :Mandatory: No
 :Default:

 May be used to enable (``1``, ``yes``) or disable (``0``, ``no``)
 automatic listing of shared libraries that are provided by the package.
 Shared libraries are added to provides list.

 .. note::

   By default automatic provides detection is enabled by rpm generator.

.. variable:: CPACK_RPM_PACKAGE_AUTOREQPROV
              CPACK_RPM_<component>_PACKAGE_AUTOREQPROV

 RPM spec autoreqprov field.

 :Mandatory: No
 :Default:

 Variable enables/disables autoreq and autoprov at the same time.
 See :variable:`CPACK_RPM_PACKAGE_AUTOREQ` and
 :variable:`CPACK_RPM_PACKAGE_AUTOPROV` for more details.

 .. note::

   By default automatic detection feature is enabled by rpm.

.. variable:: CPACK_RPM_PACKAGE_REQUIRES
              CPACK_RPM_<component>_PACKAGE_REQUIRES

 RPM spec requires field.

 :Mandatory: No
 :Default:

 May be used to set RPM dependencies (requires). Note that you must enclose
 the entire value between quotes when setting this variable, for example:

 .. code-block:: cmake

  set(CPACK_RPM_PACKAGE_REQUIRES "python >= 2.5.0, cmake >= 2.8")

 The required package list of an RPM file could be printed with::

  rpm -qp --requires file.rpm

.. variable:: CPACK_RPM_PACKAGE_CONFLICTS
              CPACK_RPM_<component>_PACKAGE_CONFLICTS

 RPM spec conflicts field.

 :Mandatory: No
 :Default:

 May be used to set negative RPM dependencies (conflicts). Note that you must
 enclose the entire value between quotes when setting this variable,
 for example:

 .. code-block:: cmake

  set(CPACK_RPM_PACKAGE_CONFLICTS "libxml2")

 The conflicting package list of an RPM file could be printed with::

  rpm -qp --conflicts file.rpm

.. variable:: CPACK_RPM_PACKAGE_REQUIRES_PRE
              CPACK_RPM_<component>_PACKAGE_REQUIRES_PRE

 .. versionadded:: 3.2

 RPM spec requires(pre) field.

 :Mandatory: No
 :Default:

 May be used to set RPM preinstall dependencies (requires(pre)). Note that
 you must enclose the entire value between quotes when setting this variable,
 for example:

 .. code-block:: cmake

  set(CPACK_RPM_PACKAGE_REQUIRES_PRE "shadow-utils, initscripts")

.. variable:: CPACK_RPM_PACKAGE_REQUIRES_POST
              CPACK_RPM_<component>_PACKAGE_REQUIRES_POST

 .. versionadded:: 3.2

 RPM spec requires(post) field.

 :Mandatory: No
 :Default:

 May be used to set RPM postinstall dependencies (requires(post)). Note that
 you must enclose the entire value between quotes when setting this variable,
 for example:

 .. code-block:: cmake

  set(CPACK_RPM_PACKAGE_REQUIRES_POST "shadow-utils, initscripts")

.. variable:: CPACK_RPM_PACKAGE_REQUIRES_POSTUN
              CPACK_RPM_<component>_PACKAGE_REQUIRES_POSTUN

 .. versionadded:: 3.2

 RPM spec requires(postun) field.

 :Mandatory: No
 :Default:

 May be used to set RPM postuninstall dependencies (requires(postun)). Note
 that you must enclose the entire value between quotes when setting this
 variable, for example:

 .. code-block:: cmake

  set(CPACK_RPM_PACKAGE_REQUIRES_POSTUN "shadow-utils, initscripts")

.. variable:: CPACK_RPM_PACKAGE_REQUIRES_PREUN
              CPACK_RPM_<component>_PACKAGE_REQUIRES_PREUN

 .. versionadded:: 3.2

 RPM spec requires(preun) field.

 :Mandatory: No
 :Default:

 May be used to set RPM preuninstall dependencies (requires(preun)). Note that
 you must enclose the entire value between quotes when setting this variable,
 for example:

 .. code-block:: cmake

  set(CPACK_RPM_PACKAGE_REQUIRES_PREUN "shadow-utils, initscripts")

.. variable:: CPACK_RPM_PACKAGE_SUGGESTS
              CPACK_RPM_<component>_PACKAGE_SUGGESTS

 RPM spec suggests field.

 :Mandatory: No
 :Default:

 May be used to set weak RPM dependencies (suggests). If ``rpmbuild`` doesn't
 support the ``Suggests`` tag, CPack will emit a warning and ignore this
 variable. Note that you must enclose the entire value between quotes when
 setting this variable.

.. variable:: CPACK_RPM_PACKAGE_RECOMMENDS
              CPACK_RPM_<component>_PACKAGE_RECOMMENDS

 .. versionadded:: 4.1

 RPM spec recommends field.

 :Mandatory: No
 :Default:

 May be used to set weak RPM dependencies (recommends). If ``rpmbuild`` doesn't
 support the ``Recommends`` tag, CPack will emit a warning and ignore this
 variable. Note that you must enclose the entire value between quotes when
 setting this variable.

.. variable:: CPACK_RPM_PACKAGE_SUPPLEMENTS
              CPACK_RPM_<component>_PACKAGE_SUPPLEMENTS

 .. versionadded:: 4.1

 RPM spec supplements field.

 :Mandatory: No
 :Default:

 May be used to set weak RPM dependencies (supplements). If ``rpmbuild`` doesn't
 support the ``Supplements`` tag, CPack will emit a warning and ignore this
 variable. Note that you must enclose the entire value between quotes when
 setting this variable.

.. variable:: CPACK_RPM_PACKAGE_ENHANCES
              CPACK_RPM_<component>_PACKAGE_ENHANCES

 .. versionadded:: 4.1

 RPM spec enhances field.

 :Mandatory: No
 :Default:

 May be used to set weak RPM dependencies (enhances). If ``rpmbuild`` doesn't
 support the ``Enhances`` tag, CPack will emit a warning and ignore this
 variable. Note that you must enclose the entire value between quotes when
 setting this variable.

.. variable:: CPACK_RPM_PACKAGE_PROVIDES
              CPACK_RPM_<component>_PACKAGE_PROVIDES

 RPM spec provides field.

 :Mandatory: No
 :Default:

 May be used to set RPM dependencies (provides). The provided package list
 of an RPM file could be printed with::

  rpm -qp --provides file.rpm

.. variable:: CPACK_RPM_PACKAGE_OBSOLETES
              CPACK_RPM_<component>_PACKAGE_OBSOLETES

 RPM spec obsoletes field.

 :Mandatory: No
 :Default:

 May be used to set RPM packages that are obsoleted by this one.

.. variable:: CPACK_RPM_PACKAGE_RELOCATABLE

 build a relocatable RPM.

 :Mandatory: No
 :Default: CPACK_PACKAGE_RELOCATABLE

 If this variable is set to TRUE or ON, the CPack RPM generator will try
 to build a relocatable RPM package. A relocatable RPM may
 be installed using::

  rpm --prefix or --relocate

 in order to install it at an alternate place see rpm(8). Note that
 currently this may fail if :variable:`CPACK_SET_DESTDIR` is set to ``ON``. If
 :variable:`CPACK_SET_DESTDIR` is set then you will get a warning message but
 if there is file installed with absolute path you'll get unexpected behavior.

.. variable:: CPACK_RPM_SPEC_INSTALL_POST

 .. deprecated:: 2.8.12 Use :variable:`CPACK_RPM_SPEC_MORE_DEFINE` instead.

 :Mandatory: No
 :Default:

 May be used to override the ``__spec_install_post`` section within the
 generated spec file.  This affects the install step during package creation,
 not during package installation.  For adding operations to be performed
 during package installation, use
 :variable:`CPACK_RPM_POST_INSTALL_SCRIPT_FILE` instead.

.. variable:: CPACK_RPM_SPEC_MORE_DEFINE

 RPM extended spec definitions lines.

 :Mandatory: No
 :Default:

 May be used to add any ``%define`` lines to the generated spec file.  An
 example of its use is to prevent stripping of executables (but note that
 this may also disable other default post install processing):

 .. code-block:: cmake

   set(CPACK_RPM_SPEC_MORE_DEFINE "%define __spec_install_post /bin/true")

.. variable:: CPACK_RPM_PACKAGE_DEBUG

 Toggle CPack RPM generator debug output.

 :Mandatory: No
 :Default:

 May be set when invoking cpack in order to trace debug information
 during CPack RPM run. For example you may launch CPack like this::

  cpack -D CPACK_RPM_PACKAGE_DEBUG=1 -G RPM

.. variable:: CPACK_RPM_USER_BINARY_SPECFILE
              CPACK_RPM_<componentName>_USER_BINARY_SPECFILE

 A user provided spec file.

 :Mandatory: No
 :Default:

 May be set by the user in order to specify a USER binary spec file
 to be used by the CPack RPM generator instead of generating the file.
 The specified file will be processed by configure_file(@ONLY).

.. variable:: CPACK_RPM_GENERATE_USER_BINARY_SPECFILE_TEMPLATE

 Spec file template.

 :Mandatory: No
 :Default:

 If set CPack will generate a template for USER specified binary
 spec file and stop with an error. For example launch CPack like this::

  cpack -D CPACK_RPM_GENERATE_USER_BINARY_SPECFILE_TEMPLATE=1 -G RPM

 The user may then use this file in order to hand-craft is own
 binary spec file which may be used with
 :variable:`CPACK_RPM_USER_BINARY_SPECFILE`.

.. variable:: CPACK_RPM_PRE_INSTALL_SCRIPT_FILE
              CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE
              CPACK_RPM_PRE_TRANS_SCRIPT_FILE

 Path to file containing pre install/uninstall/transaction script.

 :Mandatory: No
 :Default:

 May be used to embed a pre installation/uninstallation/transaction script in the spec file.
 The referred script file (or both) will be read and directly
 put after the ``%pre`` or ``%preun`` section
 If :variable:`CPACK_RPM_COMPONENT_INSTALL` is set to ON the install/uninstall/transaction
 script for each component can be overridden with
 :variable:`!CPACK_RPM_<COMPONENT>_PRE_INSTALL_SCRIPT_FILE`,
 :variable:`!CPACK_RPM_<COMPONENT>_PRE_UNINSTALL_SCRIPT_FILE`, and
 :variable:`!CPACK_RPM_<COMPONENT>_PRE_TRANS_SCRIPT_FILE`
 One may verify which scriptlet has been included with::

  rpm -qp --scripts  package.rpm

 .. versionadded:: 3.18
  The :variable:`!CPACK_RPM_PRE_TRANS_SCRIPT_FILE` variable.

.. variable:: CPACK_RPM_POST_INSTALL_SCRIPT_FILE
              CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE
              CPACK_RPM_POST_TRANS_SCRIPT_FILE

 Path to file containing post install/uninstall/transaction script.

 :Mandatory: No
 :Default:

 May be used to embed a post installation/uninstallation/transaction script in the spec file.
 The referred script file (or both) will be read and directly
 put after the ``%post`` or ``%postun`` section.
 If :variable:`CPACK_RPM_COMPONENT_INSTALL` is set to ON the install/uninstall/transaction
 script for each component can be overridden with
 :variable:`!CPACK_RPM_<COMPONENT>_POST_INSTALL_SCRIPT_FILE`,
 :variable:`!CPACK_RPM_<COMPONENT>_POST_UNINSTALL_SCRIPT_FILE`, and
 :variable:`!CPACK_RPM_<COMPONENT>_POST_TRANS_SCRIPT_FILE`
 One may verify which scriptlet has been included with::

  rpm -qp --scripts  package.rpm

 .. versionadded:: 3.18
  The :variable:`!CPACK_RPM_POST_TRANS_SCRIPT_FILE` variable.

.. variable:: CPACK_RPM_USER_FILELIST
              CPACK_RPM_<COMPONENT>_USER_FILELIST

 :Mandatory: No
 :Default:

 May be used to explicitly specify ``%(<directive>)`` file line
 in the spec file. Like ``%config(noreplace)`` or any other directive
 that be found in the ``%files`` section. Since
 the CPack RPM generator is generating the list of files (and directories) the
 user specified files of the :variable:`!CPACK_RPM_<COMPONENT>_USER_FILELIST` list will
 be removed from the generated list. If referring to directories do
 not add a trailing slash.

 .. versionadded:: 3.8
  You can have multiple directives per line, as in
  ``%attr(600,root,root) %config(noreplace)``.

.. variable:: CPACK_RPM_CHANGELOG_FILE

 RPM changelog file.

 :Mandatory: No
 :Default:

 May be used to embed a changelog in the spec file.
 The referred file will be read and directly put after the ``%changelog``
 section.

.. variable:: CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST

 list of path to be excluded.

 :Mandatory: No
 :Default:
  The following paths are excluded by default:
    - ``/etc``
    - ``/etc/init.d``
    - ``/usr``
    - ``/usr/bin``
    - ``/usr/include``
    - ``/usr/lib``
    - ``/usr/libx32``
    - ``/usr/lib64``
    - ``/usr/share``
    - ``/usr/share/aclocal``
    - ``/usr/share/doc``

 May be used to exclude path (directories or files) from the auto-generated
 list of paths discovered by CPack RPM. The default value contains a
 reasonable set of values if the variable is not defined by the user. If the
 variable is defined by the user then the CPack RPM generator will NOT any of
 the default path. If you want to add some path to the default list then you
 can use :variable:`CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION` variable.

 .. versionadded:: 3.10
  Added ``/usr/share/aclocal`` to the default list of excludes.

.. variable:: CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION

 additional list of path to be excluded.

 :Mandatory: No
 :Default:

 May be used to add more exclude path (directories or files) from the initial
 default list of excluded paths. See
 :variable:`CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST`.

.. variable:: CPACK_RPM_RELOCATION_PATHS

 .. versionadded:: 3.2

 Packages relocation paths list.

 :Mandatory: No
 :Default:

 May be used to specify more than one relocation path per relocatable RPM.
 Variable contains a list of relocation paths that if relative are prefixed
 by the value of :variable:`CPACK_RPM_<COMPONENT>_PACKAGE_PREFIX` or by the
 value of :variable:`CPACK_PACKAGING_INSTALL_PREFIX` if the component version
 is not provided.
 Variable is not component based as its content can be used to set a different
 path prefix for e.g. binary dir and documentation dir at the same time.
 Only prefixes that are required by a certain component are added to that
 component - component must contain at least one file/directory/symbolic link
 with :variable:`CPACK_RPM_RELOCATION_PATHS` prefix for a certain relocation
 path to be added. Package will not contain any relocation paths if there are
 no files/directories/symbolic links on any of the provided prefix locations.
 Packages that either do not contain any relocation paths or contain
 files/directories/symbolic links that are outside relocation paths print
 out an :command:`AUTHOR_WARNING <message>` that RPM will be partially relocatable.

.. variable:: CPACK_RPM_<COMPONENT>_PACKAGE_PREFIX

 .. versionadded:: 3.2

 Per component relocation path install prefix.

 :Mandatory: No
 :Default: :variable:`CPACK_PACKAGING_INSTALL_PREFIX`

 May be used to set per component :variable:`CPACK_PACKAGING_INSTALL_PREFIX`
 for relocatable RPM packages.

.. variable:: CPACK_RPM_NO_INSTALL_PREFIX_RELOCATION
              CPACK_RPM_NO_<COMPONENT>_INSTALL_PREFIX_RELOCATION

 .. versionadded:: 3.3

 Removal of default install prefix from relocation paths list.

 :Mandatory: No
 :Default: :variable:`CPACK_PACKAGING_INSTALL_PREFIX` or
    :variable:`CPACK_RPM_<COMPONENT>_PACKAGE_PREFIX`
    are treated as one of relocation paths

 May be used to remove :variable:`CPACK_PACKAGING_INSTALL_PREFIX` and
 :variable:`CPACK_RPM_<COMPONENT>_PACKAGE_PREFIX`
 from relocatable RPM prefix paths.

.. variable:: CPACK_RPM_ADDITIONAL_MAN_DIRS

 .. versionadded:: 3.3

 :Mandatory: No
 :Default:
  Regular expressions that are added by default were taken from ``brp-compress`` RPM macro:
    - ``/usr/man/man.*``
    - ``/usr/man/.*/man.*``
    - ``/usr/info.*``
    - ``/usr/share/man/man.*``
    - ``/usr/share/man/.*/man.*``
    - ``/usr/share/info.*``
    - ``/usr/kerberos/man.*``
    - ``/usr/X11R6/man/man.*``
    - ``/usr/lib/perl5/man/man.*``
    - ``/usr/share/doc/.*/man/man.*``
    - ``/usr/lib/.*/man/man.*``

 May be used to set additional man dirs that could potentially be compressed
 by brp-compress RPM macro. Variable content must be a list of regular
 expressions that point to directories containing man files or to man files
 directly. Note that in order to compress man pages a path must also be
 present in brp-compress RPM script and that brp-compress script must be
 added to RPM configuration by the operating system.

.. variable:: CPACK_RPM_DEFAULT_USER
              CPACK_RPM_<compName>_DEFAULT_USER

 .. versionadded:: 3.6

 default user ownership of RPM content

 :Mandatory: No
 :Default: ``root``

 Value should be user name and not UID.
 Note that ``<compName>`` must be in upper-case.

.. variable:: CPACK_RPM_DEFAULT_GROUP
              CPACK_RPM_<compName>_DEFAULT_GROUP

 .. versionadded:: 3.6

 default group ownership of RPM content

 :Mandatory: No
 :Default: root

 Value should be group name and not GID.
 Note that ``<compName>`` must be in upper-case.

.. variable:: CPACK_RPM_DEFAULT_FILE_PERMISSIONS
              CPACK_RPM_<compName>_DEFAULT_FILE_PERMISSIONS

 .. versionadded:: 3.6

 default permissions used for packaged files

 :Mandatory: No
 :Default: (system default)

 Accepted values are lists with PERMISSIONS. Valid permissions
 are:

 - ``OWNER_READ``
 - ``OWNER_WRITE``
 - ``OWNER_EXECUTE``
 - ``GROUP_READ``
 - ``GROUP_WRITE``
 - ``GROUP_EXECUTE``
 - ``WORLD_READ``
 - ``WORLD_WRITE``
 - ``WORLD_EXECUTE``

 Note that ``<compName>`` must be in upper-case.

.. variable:: CPACK_RPM_DEFAULT_DIR_PERMISSIONS
              CPACK_RPM_<compName>_DEFAULT_DIR_PERMISSIONS

 .. versionadded:: 3.6

 default permissions used for packaged directories

 :Mandatory: No
 :Default: (system default)

 Accepted values are lists with PERMISSIONS. Valid permissions
 are the same as for :variable:`CPACK_RPM_DEFAULT_FILE_PERMISSIONS`.
 Note that ``<compName>`` must be in upper-case.

.. variable:: CPACK_RPM_INSTALL_WITH_EXEC

 .. versionadded:: 3.11

 force execute permissions on programs and shared libraries

 :Mandatory: No
 :Default: (system default)

 Force set owner, group and world execute permissions on programs and shared
 libraries. This can be used for creating valid rpm packages on systems such
 as Debian where shared libraries do not have execute permissions set.

.. note::

 Programs and shared libraries without execute permissions are ignored during
 separation of debug symbols from the binary for debuginfo packages.

Packaging of Symbolic Links
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.3

The CPack RPM generator supports packaging of symbolic links:

.. code-block:: cmake

  execute_process(COMMAND ${CMAKE_COMMAND}
    -E create_symlink <relative_path_location> <symlink_name>)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/<symlink_name>
    DESTINATION <symlink_location> COMPONENT libraries)

Symbolic links will be optimized (paths will be shortened if possible)
before being added to the package or if multiple relocation paths are
detected, a post install symlink relocation script will be generated.

Symbolic links may point to locations that are not packaged by the same
package (either a different component or even not packaged at all) but
those locations will be treated as if they were a part of the package
while determining if symlink should be either created or present in a
post install script - depending on relocation paths.

.. versionchanged:: 3.6
 Symbolic links that point to locations outside packaging path produce a
 warning and are treated as non relocatable permanent symbolic links.
 Previous versions of CMake produced an error in this case.

Currently there are a few limitations though:

* For component based packaging component interdependency is not checked
  when processing symbolic links. Symbolic links pointing to content of
  a different component are treated the same way as if pointing to location
  that will not be packaged.

* Symbolic links pointing to a location through one or more intermediate
  symbolic links will not be handled differently - if the intermediate
  symbolic link(s) is also on a relocatable path, relocating it during
  package installation may cause initial symbolic link to point to an
  invalid location.

Packaging of debug information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.7

Debuginfo packages contain debug symbols and sources for debugging packaged
binaries.

Debuginfo RPM packaging has its own set of variables:

.. variable:: CPACK_RPM_DEBUGINFO_PACKAGE
              CPACK_RPM_<component>_DEBUGINFO_PACKAGE

 Enable generation of debuginfo RPM package(s).

 :Mandatory: No
 :Default: ``OFF``

.. note::

 Binaries must contain debug symbols before packaging so use either ``Debug``
 or ``RelWithDebInfo`` for :variable:`CMAKE_BUILD_TYPE` variable value.

 Additionally, if :variable:`CPACK_STRIP_FILES` is set, the files will be stripped before
 they get to the RPM generator, so will not contain debug symbols and
 a debuginfo package will not get built. Do not use with :variable:`CPACK_STRIP_FILES`.

.. note::

 Packages generated from packages without binary files, with binary files but
 without execute permissions or without debug symbols will cause packaging
 termination.

.. variable:: CPACK_BUILD_SOURCE_DIRS

 Provides locations of root directories of source files from which binaries
 were built.

 :Mandatory: Yes if :variable:`CPACK_RPM_DEBUGINFO_PACKAGE` is set
 :Default:

.. note::

 For CMake project :variable:`CPACK_BUILD_SOURCE_DIRS` is set by default to
 point to :variable:`CMAKE_SOURCE_DIR` and :variable:`CMAKE_BINARY_DIR` paths.

.. note::

 Sources with path prefixes that do not fall under any location provided with
 :variable:`CPACK_BUILD_SOURCE_DIRS` will not be present in debuginfo package.

.. variable:: CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX
              CPACK_RPM_<component>_BUILD_SOURCE_DIRS_PREFIX

 Prefix of location where sources will be placed during package installation.

 :Mandatory: Yes if :variable:`CPACK_RPM_DEBUGINFO_PACKAGE` is set
 :Default: ``/usr/src/debug/${CPACK_PACKAGE_FILE_NAME}`` and
    for component packaging ``/usr/src/debug/${CPACK_PACKAGE_FILE_NAME}-<component>``

.. note::

 Each source path prefix is additionally suffixed by ``src_<index>`` where
 index is index of the path used from :variable:`CPACK_BUILD_SOURCE_DIRS`
 variable. This produces ``${CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX}/src_<index>``
 replacement path.
 Limitation is that replaced path part must be shorter or of equal
 length than the length of its replacement. If that is not the case either
 :variable:`CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX` variable has to be set to
 a shorter path or source directories must be placed on a longer path.

.. variable:: CPACK_RPM_DEBUGINFO_EXCLUDE_DIRS

 Directories containing sources that should be excluded from debuginfo packages.

 :Mandatory: No
 :Default:
  The following paths are excluded by default:
    - ``/usr``
    - ``/usr/src``
    - ``/usr/src/debug``

 Listed paths are owned by other RPM packages and should therefore not be
 deleted on debuginfo package uninstallation.

.. variable:: CPACK_RPM_DEBUGINFO_EXCLUDE_DIRS_ADDITION

 Paths that should be appended to :variable:`CPACK_RPM_DEBUGINFO_EXCLUDE_DIRS`
 for exclusion.

 :Mandatory: No
 :Default:

.. variable:: CPACK_RPM_DEBUGINFO_SINGLE_PACKAGE

 .. versionadded:: 3.8

 Create a single debuginfo package even if components packaging is set.

 :Mandatory: No
 :Default: ``OFF``

 When this variable is enabled it produces a single debuginfo package even if
 component packaging is enabled.

 When using this feature in combination with components packaging and there is
 more than one component this variable requires :variable:`CPACK_RPM_MAIN_COMPONENT`
 to be set.

.. note::

 If none of the :variable:`CPACK_RPM_<component>_DEBUGINFO_PACKAGE` variables
 is set then :variable:`CPACK_RPM_DEBUGINFO_PACKAGE` is automatically set to
 ``ON`` when :variable:`CPACK_RPM_DEBUGINFO_SINGLE_PACKAGE` is set.

.. variable:: CPACK_RPM_DEBUGINFO_FILE_NAME
              CPACK_RPM_<component>_DEBUGINFO_FILE_NAME

 .. versionadded:: 3.9

 Debuginfo package file name.

 :Mandatory: No
 :Default: rpmbuild tool generated package file name

 Alternatively provided debuginfo package file name must end with ``.rpm``
 suffix and should differ from file names of other generated packages.

 Variable may contain ``@cpack_component@`` placeholder which will be
 replaced by component name if component packaging is enabled otherwise it
 deletes the placeholder.

 Setting the variable to ``RPM-DEFAULT`` may be used to explicitly set
 filename generation to default.

.. note::

 :variable:`CPACK_RPM_FILE_NAME` also supports rpmbuild tool generated package
 file name - disabled by default but can be enabled by setting the variable to
 ``RPM-DEFAULT``.

Packaging of sources (SRPM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.7

SRPM packaging is enabled by setting :variable:`CPACK_RPM_PACKAGE_SOURCES`
variable while usually using :variable:`CPACK_INSTALLED_DIRECTORIES` variable
to provide directory containing CMakeLists.txt and source files.

For CMake projects SRPM package would be produced by executing::

  cpack -G RPM --config ./CPackSourceConfig.cmake

.. note::

 Produced SRPM package is expected to be built with :manual:`cmake(1)` executable
 and packaged with :manual:`cpack(1)` executable so CMakeLists.txt has to be
 located in root source directory and must be able to generate binary rpm
 packages by executing :option:`cpack -G` command. The two executables as well as
 rpmbuild must also be present when generating binary rpm packages from the
 produced SRPM package.

Once the SRPM package is generated it can be used to generate binary packages
by creating a directory structure for rpm generation and executing rpmbuild
tool::

  mkdir -p build_dir/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}
  rpmbuild --define "_topdir <path_to_build_dir>" --rebuild <SRPM_file_name>

Generated packages will be located in build_dir/RPMS directory or its sub
directories.

.. note::

 SRPM package internally uses CPack/RPM generator to generate binary packages
 so CMakeScripts.txt can decide during the SRPM to binary rpm generation step
 what content the package(s) should have as well as how they should be packaged
 (monolithic or components). CMake can decide this for e.g. by reading environment
 variables set by the package manager before starting the process of generating
 binary rpm packages. This way a single SRPM package can be used to produce
 different binary rpm packages on different platforms depending on the platform's
 packaging rules.

Source RPM packaging has its own set of variables:

.. variable:: CPACK_RPM_PACKAGE_SOURCES

 Should the content be packaged as a source rpm (default is binary rpm).

 :Mandatory: No
 :Default: ``OFF``

.. note::

 For cmake projects :variable:`CPACK_RPM_PACKAGE_SOURCES` variable is set
 to ``OFF`` in CPackConfig.cmake and ``ON`` in CPackSourceConfig.cmake
 generated files.

.. variable:: CPACK_RPM_SOURCE_PKG_BUILD_PARAMS

 Additional command-line parameters provided to :manual:`cmake(1)` executable.

 :Mandatory: No
 :Default:

.. variable:: CPACK_RPM_SOURCE_PKG_PACKAGING_INSTALL_PREFIX

 Packaging install prefix that would be provided in :variable:`CPACK_PACKAGING_INSTALL_PREFIX`
 variable for producing binary RPM packages.

 :Mandatory: Yes
 :Default: ``/``

.. variable:: CPACK_RPM_BUILDREQUIRES

 List of source rpm build dependencies.

 :Mandatory: No
 :Default:

 May be used to set source RPM build dependencies (BuildRequires). Note that
 you must enclose the entire value between quotes when setting this variable,
 for example:

 .. code-block:: cmake

  set(CPACK_RPM_BUILDREQUIRES "python >= 2.5.0, cmake >= 2.8")

.. variable:: CPACK_RPM_REQUIRES_EXCLUDE_FROM

 .. versionadded:: 3.22

 :Mandatory: No
 :Default:

 May be used to keep the dependency generator from scanning specific files
 or directories for dependencies.  Note that you can use a regular
 expression that matches all of the directories or files, for example:

 .. code-block:: cmake

  set(CPACK_RPM_REQUIRES_EXCLUDE_FROM "bin/libqsqloci.*\\.so.*")
