CMAKE_CXX_MODULE_STD
--------------------

.. versionadded:: 3.30

Whether to add utility targets as dependencies to targets with at least
``cxx_std_23`` or not.

.. note::

   This setting is meaningful only when experimental support for ``import
   std;`` has been enabled by the ``CMAKE_EXPERIMENTAL_CXX_IMPORT_STD`` gate.

This variable is used to initialize the :prop_tgt:`CXX_MODULE_STD` property on
all targets.  See that target property for additional information.
