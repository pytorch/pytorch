SYSTEM
------

.. versionadded:: 3.25

This directory property is used to initialize the :prop_tgt:`SYSTEM`
target property for non-imported targets created in that directory.
It is set to true by :command:`add_subdirectory` and
:command:`FetchContent_Declare` when the ``SYSTEM`` option is given
as an argument to those commands.
