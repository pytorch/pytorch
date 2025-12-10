load_cache
----------

Load in the values from another project's ``CMakeCache.txt`` cache file.  This
is useful for projects that depend on another project built in a separate
directory tree.

This command has two signatures.  The recommended signature is:

.. signature::
  load_cache(<build-dir> READ_WITH_PREFIX <prefix> <entry>...)
  :target: READ_WITH_PREFIX

  Loads the cache file from the specified ``<build-dir>`` build directory and
  retrieves the listed cache entries.  The retrieved values are stored in local
  variables, with their names prefixed by the provided ``<prefix>``.  This only
  reads the cache values; it does not create or modify entries in the local
  project's cache.

  ``READ_WITH_PREFIX <prefix>``
    For each cache ``<entry>``, a local variable is created using the specified
    ``<prefix>`` followed by the entry name.

  This signature can be also used in :option:`cmake -P` script mode.

The following signature of this command is strongly discouraged, but it is
provided for backward compatibility:

.. signature::
  load_cache(<build-dir> [EXCLUDE <entry>...] [INCLUDE_INTERNALS <entry>...])
  :target: raw

  This form loads the cache file from the specified ``<build-dir>`` build
  directory and imports all its non-internal cache entries into the local
  project's cache as internal cache variables.  By default, only non-internal
  entries are imported, unless the ``INCLUDE_INTERNALS`` option is used.

  The options are:

  ``EXCLUDE <entry>...``
    This option can be used to exclude a given list of non-internal cache
    entries when importing values.
  ``INCLUDE_INTERNALS <entry>...``
    This option can be used to provide a list of internal cache entries to
    include in addition to the non-internal cache entries.

  This signature can be used only in CMake projects.  Script mode is not
  supported.

.. note::

  Instead of loading the outside project's cache file and manually accessing
  variables, a more robust and convenient approach is to use the
  :command:`export` command in the outside project, when available.  This allows
  the project to provide its targets, configuration, or features in a
  structured and maintainable way, making integration simpler and less
  error-prone.

Examples
^^^^^^^^

Reading specific cache variables from another project and storing them as local
variables:

.. code-block:: cmake

  load_cache(
    path/to/other-project/build-dir
    READ_WITH_PREFIX prefix_
    OTHER_PROJECT_CACHE_VAR_1
    OTHER_PROJECT_CACHE_VAR_2
  )

  message(STATUS "${prefix_OTHER_PROJECT_CACHE_VAR_1")
  message(STATUS "${prefix_OTHER_PROJECT_CACHE_VAR_2")
  # Outputs:
  # -- some-value...
  # -- another-value...

Reading all non-internal cache entries from another project and storing them as
internal cache variables using the obsolete signature:

.. code-block:: cmake

  load_cache(path/to/other-project/build-dir)

  message(STATUS "${OTHER_PROJECT_CACHE_VAR_1")
  message(STATUS "${OTHER_PROJECT_CACHE_VAR_2")
  # Outputs:
  # -- some-value...
  # -- another-value...

Excluding specific non-internal cache entries and including internal ones using
the obsolete signature:

.. code-block:: cmake

  load_cache(
    path/to/other-project/build-dir
    EXCLUDE OTHER_PROJECT_CACHE_VAR_2
    INCLUDE_INTERNALS OTHER_PROJECT_INTERNAL_CACHE_VAR
  )

  message(STATUS "${OTHER_PROJECT_CACHE_VAR_1")
  message(STATUS "${OTHER_PROJECT_CACHE_VAR_2")
  message(STATUS "${OTHER_PROJECT_INTERNAL_CACHE_VAR}")
  # Outputs:
  # -- some-value...
  # --
  # -- some-internal-value...
