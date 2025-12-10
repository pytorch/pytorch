INTERFACE_AUTOMOC_MACRO_NAMES
-----------------------------

.. versionadded:: 3.27

A :ref:`semicolon-separated list <CMake Language Lists>` of macro names for
:prop_tgt:`AUTOMOC` to be propagated to consumers.

When a target with :prop_tgt:`AUTOMOC` enabled links to a library that sets
``INTERFACE_AUTOMOC_MACRO_NAMES``, the target inherits the listed macro names
and merges them with those specified in its own :prop_tgt:`AUTOMOC_MACRO_NAMES`
property.  The target will then automatically generate MOC files for source
files that contain the inherited macro names too, not just the macro names
specified in its own :prop_tgt:`AUTOMOC_MACRO_NAMES` property.

By default ``INTERFACE_AUTOMOC_MACRO_NAMES`` is empty.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.

Example 1
^^^^^^^^^

In this example, ``myapp`` inherits the macro names ``STATIC_LIB_1`` and
``STATIC_LIB_2`` from ``static_lib``.  The ``moc`` tool will then automatically
be run on any of the ``myapp`` sources which contain ``STATIC_LIB_1`` or
``STATIC_LIB_2``.

.. code-block:: cmake

  set(AUTOMOC ON)
  add_executable(myapp main.cpp)
  target_link_libraries(myapp PRIVATE static_lib)

  add_library(static_lib STATIC static.cpp)
  set_property(TARGET static_lib PROPERTY
    INTERFACE_AUTOMOC_MACRO_NAMES "STATIC_LIB_1;STATIC_LIB_2"
  )

Example 2
^^^^^^^^^

In this example, the ``INTERFACE_AUTOMOC_MACRO_NAMES`` target property of the
various ``*_deep_lib`` libraries will propagate to ``shared_lib``,
``static_lib`` and ``interface_lib``.  Because the linking relationships are
specified as ``PUBLIC`` and ``INTERFACE``, those macro names will also further
propagate transitively up to ``app``.

.. code-block:: cmake

  set(AUTOMOC ON)

  add_library(shared_deep_lib SHARED deep_lib.cpp)
  add_library(static_deep_lib STATIC deep_lib.cpp)
  add_library(interface_deep_lib INTERFACE)

  set_property(TARGET shared_deep_lib PROPERTY
    INTERFACE_AUTOMOC_MACRO_NAMES "SHARED_LINK_LIB"
  )
  set_property(TARGET static_deep_lib PROPERTY
    INTERFACE_AUTOMOC_MACRO_NAMES "STATIC_LINK_LIB"
  )
  set_property(TARGET interface_deep_lib PROPERTY
    INTERFACE_AUTOMOC_MACRO_NAMES "INTERFACE_LINK_LIB"
  )

  add_library(shared_lib SHARED lib.cpp)
  add_library(static_lib STATIC lib.cpp)
  add_library(interface_lib INTERFACE)

  # PUBLIC and INTERFACE here ensure the macro names propagate to any
  # consumers of shared_lib, static_lib or interface_lib too
  target_link_libraries(shared_lib PUBLIC shared_deep_lib)
  target_link_libraries(static_lib PUBLIC static_deep_lib)
  target_link_libraries(interface_lib INTERFACE interface_deep_lib)

  # This consumer will receive all three of the above custom macro names as
  # transitive usage requirements
  add_executable(app main.cpp)
  target_link_libraries(app PRIVATE shared_lib static_lib interface_lib)

In the above:

* ``shared_lib`` sources will be processed by ``moc`` if they contain
  ``SHARED_LINK_LIB``.
* ``static_lib`` sources will be processed by ``moc`` if they contain
  ``STATIC_LINK_LIB``.
* ``app`` sources will be processed by ``moc`` if they contain
  ``SHARED_LINK_LIB``, ``STATIC_LINK_LIB`` or ``INTERFACE_LINK_LIB``.
