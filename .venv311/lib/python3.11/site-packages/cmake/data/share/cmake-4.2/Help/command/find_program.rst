find_program
------------

.. |FIND_XXX| replace:: find_program
.. |NAMES| replace:: NAMES name1 [name2 ...] [NAMES_PER_DIR]
.. |SEARCH_XXX| replace:: program
.. |SEARCH_XXX_DESC| replace:: program
.. |prefix_XXX_SUBDIR| replace:: ``<prefix>/[s]bin``
.. |entry_XXX_SUBDIR| replace:: ``<entry>/[s]bin``

.. |FIND_XXX_REGISTRY_VIEW_DEFAULT| replace:: ``BOTH``

.. |FIND_PACKAGE_ROOT_PREFIX_PATH_XXX| replace::
   |FIND_PACKAGE_ROOT_PREFIX_PATH_XXX_SUBDIR|
.. |CMAKE_PREFIX_PATH_XXX| replace::
   |CMAKE_PREFIX_PATH_XXX_SUBDIR|
.. |CMAKE_XXX_PATH| replace:: :variable:`CMAKE_PROGRAM_PATH`
.. |CMAKE_XXX_MAC_PATH| replace:: :variable:`CMAKE_APPBUNDLE_PATH`

.. |ENV_CMAKE_PREFIX_PATH_XXX| replace::
   |ENV_CMAKE_PREFIX_PATH_XXX_SUBDIR|
.. |ENV_CMAKE_XXX_PATH| replace:: :envvar:`CMAKE_PROGRAM_PATH`
.. |ENV_CMAKE_XXX_MAC_PATH| replace:: :envvar:`CMAKE_APPBUNDLE_PATH`

.. |SYSTEM_ENVIRONMENT_PATH_XXX| replace:: The directories in ``PATH`` itself.
.. |SYSTEM_ENVIRONMENT_PATH_WINDOWS_XXX| replace:: \

.. |CMAKE_SYSTEM_PREFIX_PATH_XXX| replace::
   |CMAKE_SYSTEM_PREFIX_PATH_XXX_SUBDIR|
.. |CMAKE_SYSTEM_XXX_PATH| replace::
   :variable:`CMAKE_SYSTEM_PROGRAM_PATH`
.. |CMAKE_SYSTEM_XXX_MAC_PATH| replace::
   :variable:`CMAKE_SYSTEM_APPBUNDLE_PATH`

.. |CMAKE_FIND_ROOT_PATH_MODE_XXX| replace::
   :variable:`CMAKE_FIND_ROOT_PATH_MODE_PROGRAM`

.. include:: include/FIND_XXX.rst

When more than one value is given to the ``NAMES`` option this command by
default will consider one name at a time and search every directory
for it.  The ``NAMES_PER_DIR`` option tells this command to consider one
directory at a time and search for all names in it.

The set of files considered to be programs is platform-specific:

* On Windows, filename suffixes are considered in order ``.com``, ``.exe``,
  and no suffix.

* On non-Windows systems, no filename suffix is considered, but files
  must have execute permission (see policy :policy:`CMP0109`).

To search for scripts, specify an extension explicitly:

.. code-block:: cmake

  if(WIN32)
    set(_script_suffix .bat)
  else()
    set(_script_suffix .sh)
  endif()

  find_program(MY_SCRIPT NAMES my_script${_script_suffix})
