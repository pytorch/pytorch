INTERFACE_CXX_MODULE_SETS
-------------------------

.. versionadded:: 3.28

Read-only list of the target's ``PUBLIC`` C++ module sets (i.e. all file sets
with the type ``CXX_MODULES``). Files listed in these C++ module sets can be
installed with :command:`install(TARGETS)` and exported with
:command:`install(EXPORT)` and :command:`export`.

C++ module sets may be defined using the :command:`target_sources` command
``FILE_SET`` option with type ``CXX_MODULES``.

See also :prop_tgt:`CXX_MODULE_SETS`.
