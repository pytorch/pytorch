CPack FreeBSD Generator
-----------------------

.. versionadded:: 3.10

The built in (binary) CPack FreeBSD (pkg) generator (Unix only)

Variables affecting the CPack FreeBSD (pkg) generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- .. versionadded:: 3.18
    :variable:`CPACK_ARCHIVE_THREADS`

Variables specific to CPack FreeBSD (pkg) generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CPack FreeBSD generator may be used to create pkg(8) packages -- these may
be used on FreeBSD, DragonflyBSD, NetBSD, OpenBSD, but also on Linux or OSX,
depending on the installed package-management tools -- using :module:`CPack`.

The CPack FreeBSD generator is a :module:`CPack` generator and uses the
:variable:`!CPACK_XXX` variables used by :module:`CPack`. It tries to reuse packaging
information that may already be specified for Debian packages for the
:cpack_gen:`CPack DEB Generator`. It also tries to reuse RPM packaging
information when Debian does not specify.

The CPack FreeBSD generator should work on any host with libpkg installed. The
packages it produces are specific to the host architecture and ABI.

The CPack FreeBSD generator sets package-metadata through
:variable:`!CPACK_FREEBSD_XXX` variables. The CPack FreeBSD generator, unlike the
CPack Deb generator, does not specially support componentized packages; a
single package is created from all the software artifacts created through
CMake.

All of the variables can be set specifically for FreeBSD packaging in
the CPackConfig file or in CMakeLists.txt, but most of them have defaults
that use general settings (e.g. :variable:`CMAKE_PROJECT_NAME`) or Debian-specific
variables when those make sense (e.g. the homepage of an upstream project
is usually unchanged by the flavor of packaging). When there is no Debian
information to fall back on, but the RPM packaging has it, fall back to
the RPM information (e.g. package license).

.. variable:: CPACK_FREEBSD_PACKAGE_NAME

  Sets the package name (in the package manifest, but also affects the
  output filename).

  :Mandatory: Yes
  :Default:

    - :variable:`CPACK_PACKAGE_NAME` (this is always set by CPack itself,
      based on CMAKE_PROJECT_NAME).

.. variable:: CPACK_FREEBSD_PACKAGE_COMMENT

  Sets the package comment. This is the short description displayed by
  pkg(8) in standard "pkg info" output.

  :Mandatory: Yes
  :Default:

    - :variable:`CPACK_PACKAGE_DESCRIPTION_SUMMARY` (this is always set
      by CPack itself, if nothing else sets it explicitly).

.. variable:: CPACK_FREEBSD_PACKAGE_DESCRIPTION

  Sets the package description. This is the long description of the package,
  given by "pkg info" with a specific package as argument.

  :Mandatory: Yes
  :Default:

    - :variable:`CPACK_DEBIAN_PACKAGE_DESCRIPTION` (this may be set already
      for Debian packaging, so it is used as a fallback).
    - :variable:`CPACK_PACKAGE_DESCRIPTION_SUMMARY` (this is always set
      by CPack itself, if nothing else sets it explicitly).
    - :variable:`PROJECT_DESCRIPTION` (this can be set with the ``DESCRIPTION``
      parameter for :command:`project`).

.. variable:: CPACK_FREEBSD_PACKAGE_WWW

  The URL of the web site for this package, preferably (when applicable) the
  site from which the original source can be obtained and any additional
  upstream documentation or information may be found.

  :Mandatory: Yes
  :Default:

   - :variable:`CPACK_PACKAGE_HOMEPAGE_URL`, or if that is not set,
   - :variable:`CPACK_DEBIAN_PACKAGE_HOMEPAGE` (this may be set already
     for Debian packaging, so it is used as a fallback).

  .. versionadded:: 3.12
    The :variable:`!CPACK_PACKAGE_HOMEPAGE_URL` variable.

.. variable:: CPACK_FREEBSD_PACKAGE_LICENSE

  The license, or licenses, which apply to this software package. This must
  be one or more license-identifiers that pkg recognizes as acceptable license
  identifiers (e.g. "GPLv2").

  :Mandatory: Yes
  :Default:

    - :variable:`CPACK_RPM_PACKAGE_LICENSE`

.. variable:: CPACK_FREEBSD_PACKAGE_LICENSE_LOGIC

  This variable is only of importance if there is more than one license.
  The default is "single", which is only applicable to a single license.
  Other acceptable values are determined by pkg -- those are "dual" or "multi" --
  meaning choice (OR) or simultaneous (AND) application of the licenses.

  :Mandatory: No
  :Default: single

.. variable:: CPACK_FREEBSD_PACKAGE_MAINTAINER

  The FreeBSD maintainer (e.g. ``kde@freebsd.org``) of this package.

  :Mandatory: Yes
  :Default: none

.. variable:: CPACK_FREEBSD_PACKAGE_ORIGIN

  The origin (ports label) of this package; for packages built by CPack
  outside of the ports system this is of less importance. The default
  puts the package somewhere under ``misc/``, as a stopgap.

  :Mandatory: Yes
  :Default: ``misc/<package name>``

.. variable:: CPACK_FREEBSD_PACKAGE_CATEGORIES

  The ports categories where this package lives (if it were to be built
  from ports). If none is set a single category is determined based on
  the package origin.

  :Mandatory: Yes
  :Default: derived from ``ORIGIN``

.. variable:: CPACK_FREEBSD_PACKAGE_DEPS

  A list of package origins that should be added as package dependencies.
  These are in the form ``<category>/<packagename>``, e.g. ``x11/libkonq``.
  No version information needs to be provided (this is not included
  in the manifest).

  :Mandatory: No
  :Default: empty
