A short-hand signature is:

.. parsed-literal::

   |FIND_XXX| (<VAR> name1 [path1 path2 ...])

The general signature is:

.. parsed-literal::

   |FIND_XXX| (
             <VAR>
             name | |NAMES|
             [HINTS [path | ENV var]...]
             [PATHS [path | ENV var]...]
             [REGISTRY_VIEW (64|32|64_32|32_64|HOST|TARGET|BOTH)]
             [PATH_SUFFIXES suffix1 [suffix2 ...]]
             [VALIDATOR function]
             [DOC "cache documentation string"]
             [NO_CACHE]
             [REQUIRED|OPTIONAL]
             [NO_DEFAULT_PATH]
             [NO_PACKAGE_ROOT_PATH]
             [NO_CMAKE_PATH]
             [NO_CMAKE_ENVIRONMENT_PATH]
             [NO_SYSTEM_ENVIRONMENT_PATH]
             [NO_CMAKE_SYSTEM_PATH]
             [NO_CMAKE_INSTALL_PREFIX]
             [CMAKE_FIND_ROOT_PATH_BOTH |
              ONLY_CMAKE_FIND_ROOT_PATH |
              NO_CMAKE_FIND_ROOT_PATH]
            )

This command is used to find a |SEARCH_XXX_DESC|.

Prior to searching, |FIND_XXX| checks if variable ``<VAR>`` is defined. If
the variable is not defined, the search will be performed. If the variable is
defined and its value is ``NOTFOUND``, or ends in ``-NOTFOUND``, the search
will be performed. If the variable contains any other value the search is not
performed.

  .. note::
      ``VAR`` is considered defined if it is available in the current scope. See
      the :ref:`cmake-language(7) variables <CMake Language Variables>`
      documentation for details on scopes, and the interaction of normal
      variables and cache entries.

The results of the search will be stored in a cache entry named ``<VAR>``.
Future calls to |FIND_XXX| will inspect this cache entry when specifying the
same ``<VAR>``. This optimization ensures successful searches will not be
repeated unless the cache entry is :command:`unset`.

If the |SEARCH_XXX| is found the recorded value in cache entry ``<VAR>`` will
be the result of the search. If nothing is found, the recorded value will be
``<VAR>-NOTFOUND``.

Options include:

``NAMES``
  Specify one or more possible names for the |SEARCH_XXX|.

  When using this to specify names with and without a version
  suffix, we recommend specifying the unversioned name first
  so that locally-built packages can be found before those
  provided by distributions.

``HINTS``, ``PATHS``
  Specify directories to search in addition to the default locations.
  The ``ENV var`` sub-option reads paths from a system environment
  variable.

  .. versionchanged:: 3.24
    On ``Windows`` platform, it is possible to include registry queries as part
    of the directories, using a :ref:`dedicated syntax <Find Using Windows Registry>`.
    Such specifications will be ignored on all other platforms.

``REGISTRY_VIEW``
  .. versionadded:: 3.24

  .. include:: include/FIND_XXX_REGISTRY_VIEW.rst

``PATH_SUFFIXES``
  Specify additional subdirectories to check below each directory
  location otherwise considered.

``VALIDATOR``
  .. versionadded:: 3.25

  Specify a :command:`function` to be called for each candidate item found
  (a :command:`macro` cannot be provided, that will result in an error).
  Two arguments will be passed to the validator function: the name of a
  result variable, and the absolute path to the candidate item.  The item
  will be accepted and the search will end unless the function sets the
  value in the result variable to false in the calling scope.  The result
  variable will hold a true value when the validator function is entered.

  .. parsed-literal::

     function(my_check validator_result_var item)
       if(NOT item MATCHES ...)
         set(${validator_result_var} FALSE PARENT_SCOPE)
       endif()
     endfunction()

     |FIND_XXX| (result NAMES ... VALIDATOR my_check)

  Note that if a cached result is used, the search is skipped and any
  ``VALIDATOR`` is ignored.  The cached result is not required to pass the
  validation function.

``DOC``
  Specify the documentation string for the ``<VAR>`` cache entry.

``NO_CACHE``
  .. versionadded:: 3.21

  The result of the search will be stored in a normal variable rather than
  a cache entry.

  .. note::

    |FIND_XXX| will still check for ``<VAR>`` as usual, checking first for a
    variable, and then a cache entry. If either indicate a previous successful
    search, the search will not be performed.

  .. warning::

    This option should be used with caution because it can greatly increase
    the cost of repeated configure steps.

``REQUIRED``
  .. versionadded:: 3.18

  Stop processing with an error message if nothing is found, otherwise
  the search will be attempted again the next time |FIND_XXX| is invoked
  with the same variable.

  .. versionadded:: 4.1

    Every |FIND_XXX| command will be treated as ``REQUIRED`` when the
    :variable:`CMAKE_FIND_REQUIRED` variable is enabled.

``OPTIONAL``
  .. versionadded:: 4.1

  Ignore the value of :variable:`CMAKE_FIND_REQUIRED` and
  continue without an error message if nothing is found.
  Incompatible with ``REQUIRED``.

If ``NO_DEFAULT_PATH`` is specified, then no additional paths are
added to the search.
If ``NO_DEFAULT_PATH`` is not specified, the search process is as follows:

.. |FIND_PACKAGE_ROOT_PREFIX_PATH_XXX_SUBDIR| replace::
   |prefix_XXX_SUBDIR| for each ``<prefix>`` in the
   :variable:`<PackageName>_ROOT` CMake variable and the
   :envvar:`<PackageName>_ROOT` environment variable if
   called from within a find module loaded by
   :command:`find_package(<PackageName>)`

.. |CMAKE_PREFIX_PATH_XXX_SUBDIR| replace::
   |prefix_XXX_SUBDIR| for each ``<prefix>`` in :variable:`CMAKE_PREFIX_PATH`

.. |ENV_CMAKE_PREFIX_PATH_XXX_SUBDIR| replace::
   |prefix_XXX_SUBDIR| for each ``<prefix>`` in :envvar:`CMAKE_PREFIX_PATH`

.. |SYSTEM_ENVIRONMENT_PREFIX_PATH_XXX_SUBDIR| replace::
   |prefix_XXX_SUBDIR| for each ``<prefix>/[s]bin`` in ``PATH``, and
   |entry_XXX_SUBDIR| for other entries in ``PATH``

.. |CMAKE_SYSTEM_PREFIX_PATH_XXX_SUBDIR| replace::
   |prefix_XXX_SUBDIR| for each ``<prefix>`` in
   :variable:`CMAKE_SYSTEM_PREFIX_PATH`

1. If called from within a find module or any other script loaded by a call to
   :command:`find_package(<PackageName>)`, search prefixes unique to the
   current package being found.  See policy :policy:`CMP0074`.

   .. versionadded:: 3.12

   Specifically, search paths specified by the following variables, in order:

   a. :variable:`<PackageName>_ROOT` CMake variable,
      where ``<PackageName>`` is the case-preserved package name.

   b. :variable:`<PACKAGENAME>_ROOT` CMake variable,
      where ``<PACKAGENAME>`` is the upper-cased package name.
      See policy :policy:`CMP0144`.

      .. versionadded:: 3.27

   c. :envvar:`<PackageName>_ROOT` environment variable,
      where ``<PackageName>`` is the case-preserved package name.

   d. :envvar:`<PACKAGENAME>_ROOT` environment variable,
      where ``<PACKAGENAME>`` is the upper-cased package name.
      See policy :policy:`CMP0144`.

      .. versionadded:: 3.27

   The package root variables are maintained as a stack, so if called from
   nested find modules or config packages, root paths from the parent's find
   module or config package will be searched after paths from the current
   module or package.  In other words, the search order would be
   ``<CurrentPackage>_ROOT``, ``ENV{<CurrentPackage>_ROOT}``,
   ``<ParentPackage>_ROOT``, ``ENV{<ParentPackage>_ROOT}``, etc.
   This can be skipped if ``NO_PACKAGE_ROOT_PATH`` is passed or by setting
   the :variable:`CMAKE_FIND_USE_PACKAGE_ROOT_PATH` to ``FALSE``.

   * |FIND_PACKAGE_ROOT_PREFIX_PATH_XXX|

2. Search paths specified in cmake-specific cache variables.
   These are intended to be used on the command line with a ``-DVAR=value``.
   The values are interpreted as :ref:`semicolon-separated lists <CMake Language Lists>`.
   This can be skipped if ``NO_CMAKE_PATH`` is passed or by setting the
   :variable:`CMAKE_FIND_USE_CMAKE_PATH` to ``FALSE``.

   * |CMAKE_PREFIX_PATH_XXX|
   * |CMAKE_XXX_PATH|
   * |CMAKE_XXX_MAC_PATH|

3. Search paths specified in cmake-specific environment variables.
   These are intended to be set in the user's shell configuration,
   and therefore use the host's native path separator
   (``;`` on Windows and ``:`` on UNIX).
   This can be skipped if ``NO_CMAKE_ENVIRONMENT_PATH`` is passed or
   by setting the :variable:`CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH` to ``FALSE``.

   * |ENV_CMAKE_PREFIX_PATH_XXX|
   * |ENV_CMAKE_XXX_PATH|
   * |ENV_CMAKE_XXX_MAC_PATH|

4. Search the paths specified by the ``HINTS`` option.
   These should be paths computed by system introspection, such as a
   hint provided by the location of another item already found.
   Hard-coded guesses should be specified with the ``PATHS`` option.

5. Search the standard system environment variables.
   This can be skipped if ``NO_SYSTEM_ENVIRONMENT_PATH`` is passed or by
   setting the :variable:`CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH` to ``FALSE``.

   * |SYSTEM_ENVIRONMENT_PATH_XXX|

   |SYSTEM_ENVIRONMENT_PATH_WINDOWS_XXX|

6. Search cmake variables defined in the Platform files
   for the current system.  The searching of ``CMAKE_INSTALL_PREFIX`` and
   ``CMAKE_STAGING_PREFIX`` can be
   skipped if ``NO_CMAKE_INSTALL_PREFIX`` is passed or by setting the
   :variable:`CMAKE_FIND_USE_INSTALL_PREFIX` to ``FALSE``. All these locations
   can be skipped if ``NO_CMAKE_SYSTEM_PATH`` is passed or by setting the
   :variable:`CMAKE_FIND_USE_CMAKE_SYSTEM_PATH` to ``FALSE``.

   * |CMAKE_SYSTEM_PREFIX_PATH_XXX|
   * |CMAKE_SYSTEM_XXX_PATH|
   * |CMAKE_SYSTEM_XXX_MAC_PATH|

   The platform paths that these variables contain are locations that
   typically include installed software. An example being ``/usr/local`` for
   UNIX based platforms.

7. Search the paths specified by the PATHS option
   or in the short-hand version of the command.
   These are typically hard-coded guesses.

The :variable:`CMAKE_IGNORE_PATH`, :variable:`CMAKE_IGNORE_PREFIX_PATH`,
:variable:`CMAKE_SYSTEM_IGNORE_PATH` and
:variable:`CMAKE_SYSTEM_IGNORE_PREFIX_PATH` variables can also cause some
of the above locations to be ignored.

.. versionadded:: 3.16
  Added ``CMAKE_FIND_USE_<CATEGORY>_PATH`` variables to globally disable
  various search locations.

.. |FIND_ARGS_XXX| replace:: <VAR> NAMES name

On macOS the :variable:`CMAKE_FIND_FRAMEWORK` and
:variable:`CMAKE_FIND_APPBUNDLE` variables determine the order of
preference between Apple-style and unix-style package components.

.. include:: include/FIND_XXX_ROOT.rst
.. include:: include/FIND_XXX_ORDER.rst
