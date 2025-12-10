CPack Archive Generator
-----------------------

CPack generator for packaging files into an archive, which can have
any of the following formats:

- 7Z - 7zip - (``.7z``)

  .. versionadded:: 3.1

- TAR (``.tar``)

  .. versionadded:: 4.0

- TBZ2 (``.tar.bz2``)

- TGZ (``.tar.gz``)

- TXZ (``.tar.xz``)

  .. versionadded:: 3.1

- TZ (``.tar.Z``)

- TZST (``.tar.zst``)

  .. versionadded:: 3.16

- ZIP (``.zip``)

When this generator is called from ``CPackSourceConfig.cmake`` (or through
the ``package_source`` target), then the generated archive will contain all
files in the project directory, except those specified in
:variable:`CPACK_SOURCE_IGNORE_FILES`.  The following is one example of
packaging all source files of a project:

.. code-block:: cmake

  set(CPACK_SOURCE_GENERATOR "TGZ")
  set(CPACK_SOURCE_IGNORE_FILES
    \\.git/
    build/
    ".*~$"
  )
  set(CPACK_VERBATIM_VARIABLES YES)
  include(CPack)

When this generator is called from ``CPackConfig.cmake`` (or through the
``package`` target), then the generated archive will contain all files
that have been installed via CMake's :command:`install` command (and the
deprecated commands :command:`install_files`, :command:`install_programs`,
and :command:`install_targets`).

Variables specific to CPack Archive generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. variable:: CPACK_ARCHIVE_FILE_NAME

  .. versionadded:: 3.9

  Archive name for component-based packages, without extension.

  :Default: :variable:`CPACK_PACKAGE_FILE_NAME`

  The extension is appended automatically.

  If :variable:`CPACK_COMPONENTS_GROUPING` is set to ``ALL_COMPONENTS_IN_ONE``,
  this will be the name of the one output archive.

  .. versionchanged:: 4.0

    This variable also works for non-component packages.

.. variable:: CPACK_ARCHIVE_<component>_FILE_NAME

  .. versionadded:: 3.9

  Component archive name without extension.

  :Default: ``<CPACK_ARCHIVE_FILE_NAME>-<component>``, with spaces replaced
    by ``'-'``.

  The extension is appended automatically. Note that ``<component>`` is all
  uppercase in the variable name.

.. variable:: CPACK_ARCHIVE_FILE_EXTENSION

  .. versionadded:: 3.25

  Archive file extension.

  :Default: Default values are given in the list above.

.. variable:: CPACK_ARCHIVE_COMPONENT_INSTALL

  Enable component packaging.

  :Default: ``OFF``

  If enabled (``ON``) multiple packages are generated. By default a single package
  containing files of all components is generated.

Variables used by CPack Archive generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These variables are used by the Archive generator, but are also available to
CPack generators which are essentially archives at their core. These include:

- :cpack_gen:`CPack Cygwin Generator`
- :cpack_gen:`CPack FreeBSD Generator`

.. variable:: CPACK_ARCHIVE_THREADS

  .. versionadded:: 3.18

  The number of threads to use when performing the compression.

  :Default: value of :variable:`CPACK_THREADS`

  If set to ``0``, the number of available cores on the machine will be used instead.
  Note that not all compression modes support threading in all environments.

  .. versionadded:: 3.21

    Official CMake binaries available on ``cmake.org`` now ship
    with a ``liblzma`` that supports parallel compression.
    Older versions did not.
