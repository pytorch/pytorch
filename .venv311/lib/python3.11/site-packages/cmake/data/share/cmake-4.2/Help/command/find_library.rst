find_library
------------

.. |FIND_XXX| replace:: find_library
.. |NAMES| replace:: NAMES name1 [name2 ...] [NAMES_PER_DIR]
.. |SEARCH_XXX| replace:: library
.. |SEARCH_XXX_DESC| replace:: library
.. |prefix_XXX_SUBDIR| replace:: ``<prefix>/lib``
.. |entry_XXX_SUBDIR| replace:: ``<entry>/lib``

.. |FIND_XXX_REGISTRY_VIEW_DEFAULT| replace:: ``TARGET``

.. |FIND_PACKAGE_ROOT_PREFIX_PATH_XXX| replace::
   ``<prefix>/lib/<arch>`` if :variable:`CMAKE_LIBRARY_ARCHITECTURE` is set,
   and |FIND_PACKAGE_ROOT_PREFIX_PATH_XXX_SUBDIR|
.. |CMAKE_PREFIX_PATH_XXX| replace::
   ``<prefix>/lib/<arch>`` if :variable:`CMAKE_LIBRARY_ARCHITECTURE` is set,
   and |CMAKE_PREFIX_PATH_XXX_SUBDIR|
.. |CMAKE_XXX_PATH| replace:: :variable:`CMAKE_LIBRARY_PATH`
.. |CMAKE_XXX_MAC_PATH| replace:: :variable:`CMAKE_FRAMEWORK_PATH`

.. |ENV_CMAKE_PREFIX_PATH_XXX| replace::
   ``<prefix>/lib/<arch>`` if :variable:`CMAKE_LIBRARY_ARCHITECTURE` is set,
   and |ENV_CMAKE_PREFIX_PATH_XXX_SUBDIR|
.. |ENV_CMAKE_XXX_PATH| replace:: :envvar:`CMAKE_LIBRARY_PATH`
.. |ENV_CMAKE_XXX_MAC_PATH| replace:: :envvar:`CMAKE_FRAMEWORK_PATH`

.. |SYSTEM_ENVIRONMENT_PATH_XXX| replace:: The directories in ``LIB``
   and ``PATH``.
.. |SYSTEM_ENVIRONMENT_PATH_WINDOWS_XXX| replace::
   On Windows hosts, CMake 3.3 through 3.27 searched additional paths:
   ``<prefix>/lib/<arch>`` if :variable:`CMAKE_LIBRARY_ARCHITECTURE`
   is set, and |SYSTEM_ENVIRONMENT_PREFIX_PATH_XXX_SUBDIR|.
   This behavior was removed by CMake 3.28.

.. |CMAKE_SYSTEM_PREFIX_PATH_XXX| replace::
   ``<prefix>/lib/<arch>`` if :variable:`CMAKE_LIBRARY_ARCHITECTURE` is set,
   and |CMAKE_SYSTEM_PREFIX_PATH_XXX_SUBDIR|
.. |CMAKE_SYSTEM_XXX_PATH| replace::
   :variable:`CMAKE_SYSTEM_LIBRARY_PATH`
.. |CMAKE_SYSTEM_XXX_MAC_PATH| replace::
   :variable:`CMAKE_SYSTEM_FRAMEWORK_PATH`

.. |CMAKE_FIND_ROOT_PATH_MODE_XXX| replace::
   :variable:`CMAKE_FIND_ROOT_PATH_MODE_LIBRARY`

.. include:: include/FIND_XXX.rst

When more than one value is given to the ``NAMES`` option this command by
default will consider one name at a time and search every directory
for it.  The ``NAMES_PER_DIR`` option tells this command to consider one
directory at a time and search for all names in it.

Each library name given to the ``NAMES`` option is first considered
as is, if it contains a library suffix, and then considered with
platform-specific prefixes (e.g. ``lib``) and suffixes (e.g. ``.so``),
as defined by the variables :variable:`CMAKE_FIND_LIBRARY_PREFIXES` and
:variable:`CMAKE_FIND_LIBRARY_SUFFIXES`. Therefore one
may specify library file names such as ``libfoo.a`` directly.
This can be used to locate static libraries on UNIX-like systems.

If the library found is a framework, then ``<VAR>`` will be set to the full
path to the framework ``<fullPath>/A.framework``.  When a full path to a
framework is used as a library, CMake will use a ``-framework A``, and a
``-F<fullPath>`` to link the framework to the target.

.. versionadded:: 3.28

  The library found can now be a ``.xcframework`` folder.

If the :variable:`CMAKE_FIND_LIBRARY_CUSTOM_LIB_SUFFIX` variable is set all
search paths will be tested as normal, with the suffix appended, and with
all matches of ``lib/`` replaced with
``lib${CMAKE_FIND_LIBRARY_CUSTOM_LIB_SUFFIX}/``.  This variable overrides
the :prop_gbl:`FIND_LIBRARY_USE_LIB32_PATHS`,
:prop_gbl:`FIND_LIBRARY_USE_LIBX32_PATHS`,
and :prop_gbl:`FIND_LIBRARY_USE_LIB64_PATHS` global properties.

If the :prop_gbl:`FIND_LIBRARY_USE_LIB32_PATHS` global property is set
all search paths will be tested as normal, with ``32/`` appended, and
with all matches of ``lib/`` replaced with ``lib32/``.  This property is
automatically set for the platforms that are known to need it if at
least one of the languages supported by the :command:`project` command
is enabled.

If the :prop_gbl:`FIND_LIBRARY_USE_LIBX32_PATHS` global property is set
all search paths will be tested as normal, with ``x32/`` appended, and
with all matches of ``lib/`` replaced with ``libx32/``.  This property is
automatically set for the platforms that are known to need it if at
least one of the languages supported by the :command:`project` command
is enabled.

If the :prop_gbl:`FIND_LIBRARY_USE_LIB64_PATHS` global property is set
all search paths will be tested as normal, with ``64/`` appended, and
with all matches of ``lib/`` replaced with ``lib64/``.  This property is
automatically set for the platforms that are known to need it if at
least one of the languages supported by the :command:`project` command
is enabled.
