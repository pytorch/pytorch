find_path
---------

.. |FIND_XXX| replace:: find_path
.. |NAMES| replace:: NAMES name1 [name2 ...]
.. |SEARCH_XXX| replace:: file in a directory
.. |SEARCH_XXX_DESC| replace:: directory containing the named file
.. |prefix_XXX_SUBDIR| replace:: ``<prefix>/include``
.. |entry_XXX_SUBDIR| replace:: ``<entry>/include``

.. |FIND_XXX_REGISTRY_VIEW_DEFAULT| replace:: ``TARGET``

.. |FIND_PACKAGE_ROOT_PREFIX_PATH_XXX| replace::
   ``<prefix>/include/<arch>`` if :variable:`CMAKE_LIBRARY_ARCHITECTURE`
   is set, and |FIND_PACKAGE_ROOT_PREFIX_PATH_XXX_SUBDIR|
.. |CMAKE_PREFIX_PATH_XXX| replace::
   ``<prefix>/include/<arch>`` if :variable:`CMAKE_LIBRARY_ARCHITECTURE`
   is set, and |CMAKE_PREFIX_PATH_XXX_SUBDIR|
.. |CMAKE_XXX_PATH| replace:: :variable:`CMAKE_INCLUDE_PATH`
.. |CMAKE_XXX_MAC_PATH| replace:: :variable:`CMAKE_FRAMEWORK_PATH`

.. |ENV_CMAKE_PREFIX_PATH_XXX| replace::
   ``<prefix>/include/<arch>`` if :variable:`CMAKE_LIBRARY_ARCHITECTURE` is set,
   and |ENV_CMAKE_PREFIX_PATH_XXX_SUBDIR|
.. |ENV_CMAKE_XXX_PATH| replace:: :envvar:`CMAKE_INCLUDE_PATH`
.. |ENV_CMAKE_XXX_MAC_PATH| replace:: :envvar:`CMAKE_FRAMEWORK_PATH`

.. |SYSTEM_ENVIRONMENT_PATH_XXX| replace:: The directories in ``INCLUDE``
   and ``PATH``.
.. |SYSTEM_ENVIRONMENT_PATH_WINDOWS_XXX| replace::
   On Windows hosts, CMake 3.3 through 3.27 searched additional paths:
   ``<prefix>/include/<arch>`` if :variable:`CMAKE_LIBRARY_ARCHITECTURE`
   is set, and |SYSTEM_ENVIRONMENT_PREFIX_PATH_XXX_SUBDIR|.
   This behavior was removed by CMake 3.28.

.. |CMAKE_SYSTEM_PREFIX_PATH_XXX| replace::
   ``<prefix>/include/<arch>`` if :variable:`CMAKE_LIBRARY_ARCHITECTURE`
   is set, and |CMAKE_SYSTEM_PREFIX_PATH_XXX_SUBDIR|
.. |CMAKE_SYSTEM_XXX_PATH| replace::
   :variable:`CMAKE_SYSTEM_INCLUDE_PATH`
.. |CMAKE_SYSTEM_XXX_MAC_PATH| replace::
   :variable:`CMAKE_SYSTEM_FRAMEWORK_PATH`

.. |CMAKE_FIND_ROOT_PATH_MODE_XXX| replace::
   :variable:`CMAKE_FIND_ROOT_PATH_MODE_INCLUDE`

.. include:: include/FIND_XXX.rst

When searching for frameworks, if the file is specified as ``A/b.h``, then
the framework search will look for ``A.framework/Headers/b.h``.  If that
is found the path will be set to the path to the framework.  CMake
will convert this to the correct ``-F`` option to include the file.
