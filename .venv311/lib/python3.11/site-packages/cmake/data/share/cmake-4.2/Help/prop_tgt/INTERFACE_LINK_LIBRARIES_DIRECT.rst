INTERFACE_LINK_LIBRARIES_DIRECT
-------------------------------

.. versionadded:: 3.24

List of libraries that consumers of this library should treat
as direct link dependencies.

This target property may be set to *include* items in a dependent
target's final set of direct link dependencies.  See the
:prop_tgt:`INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE` target property
to exclude items.

The initial set of a dependent target's direct link dependencies is
specified by its :prop_tgt:`LINK_LIBRARIES` target property.  Indirect
link dependencies are specified by the transitive closure of the direct
link dependencies' :prop_tgt:`INTERFACE_LINK_LIBRARIES` properties.
Any link dependency may specify additional direct link dependencies
using the ``INTERFACE_LINK_LIBRARIES_DIRECT`` target property.
The set of direct link dependencies is then filtered to exclude items named
by any dependency's :prop_tgt:`INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE`
target property.

.. |INTERFACE_PROPERTY_LINK_DIRECT| replace:: ``INTERFACE_LINK_LIBRARIES_DIRECT``
.. include:: include/INTERFACE_LINK_LIBRARIES_DIRECT.rst

Direct Link Dependencies as Usage Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``INTERFACE_LINK_LIBRARIES_DIRECT`` and
``INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE`` target properties
are :ref:`usage requirements <Target Usage Requirements>`.
Their effects propagate to dependent targets transitively, and can
therefore affect the direct link dependencies of every target in a
chain of dependent libraries.  Whenever some library target ``X`` links
to another library target ``Y`` whose direct or transitive usage
requirements contain ``INTERFACE_LINK_LIBRARIES_DIRECT`` or
``INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE``, the properties may affect
``X``'s list of direct link dependencies:

* If ``X`` is a shared library or executable, its dependencies are linked.
  They also affect the usage requirements with which ``X``'s sources are
  compiled.

* If ``X`` is a static library or object library, it does not actually
  link, so its dependencies at most affect the usage requirements with
  which ``X``'s sources are compiled.

The properties may also affect the list of direct link dependencies
on ``X``'s dependents:

* If ``X`` links ``Y`` publicly:

  .. code-block:: cmake

    target_link_libraries(X PUBLIC Y)

  then ``Y`` is placed in ``X``'s :prop_tgt:`INTERFACE_LINK_LIBRARIES`,
  so ``Y``'s usage requirements, including ``INTERFACE_LINK_LIBRARIES_DIRECT``,
  ``INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE``, and the usage requirements
  declared by the direct link dependencies they add, are propagated to
  ``X``'s dependents.

* If ``X`` is a static library or object library, and links ``Y`` privately:

  .. code-block:: cmake

    target_link_libraries(X PRIVATE Y)

  then ``$<LINK_ONLY:Y>`` is placed in ``X``'s
  :prop_tgt:`INTERFACE_LINK_LIBRARIES`.  ``Y``'s linking requirements,
  including ``INTERFACE_LINK_LIBRARIES_DIRECT``,
  ``INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE``, and the transitive link
  dependencies declared by the direct link dependencies they add, are
  propagated to ``X``'s dependents.  However, ``Y``'s non-linking
  usage requirements are blocked by the :genex:`LINK_ONLY` generator
  expression, and are not propagated to ``X``'s dependents.

* If ``X`` is a shared library or executable, and links ``Y`` privately:

  .. code-block:: cmake

    target_link_libraries(X PRIVATE Y)

  then ``Y`` is not placed in ``X``'s :prop_tgt:`INTERFACE_LINK_LIBRARIES`,
  so ``Y``'s usage requirements, even ``INTERFACE_LINK_LIBRARIES_DIRECT``
  and ``INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE``, are not propagated to
  ``X``'s dependents.

* In all cases, the content of ``X``'s :prop_tgt:`INTERFACE_LINK_LIBRARIES`
  is not affected by ``Y``'s ``INTERFACE_LINK_LIBRARIES_DIRECT`` or
  ``INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE``.

One may limit the effects of ``INTERFACE_LINK_LIBRARIES_DIRECT`` and
``INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE`` to a subset of dependent
targets by using the :genex:`TARGET_PROPERTY` generator expression.
For example, to limit the effects to executable targets, use an
entry of the form:

.. code-block:: cmake

  "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:...>"

Similarly, to limit the effects to specific targets, use an entry
of the form:

.. code-block:: cmake

  "$<$<BOOL:$<TARGET_PROPERTY:USE_IT>>:...>"

This entry will only affect targets that set their ``USE_IT``
target property to a true value.

Direct Link Dependency Ordering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The list of direct link dependencies for a target is computed from an
initial ordered list in its :prop_tgt:`LINK_LIBRARIES` target property.
For each item, additional direct link dependencies are discovered from
its direct and transitive ``INTERFACE_LINK_LIBRARIES_DIRECT`` usage
requirements.  Each discovered item is injected before the item that
specified it.  However, a discovered item is added at most once,
and only if it did not appear anywhere in the initial list.
This gives :prop_tgt:`LINK_LIBRARIES` control over ordering of
those direct link dependencies that it explicitly specifies.

Once all direct link dependencies have been collected, items named by
all of their :prop_tgt:`INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE`
usage requirements are removed from the final list.  This does not
affect the order of the items that remain.

Example: Static Plugins
^^^^^^^^^^^^^^^^^^^^^^^

Consider a static library ``Foo`` that provides a static plugin
``FooPlugin`` to consuming application executables, where the
implementation of the plugin depends on ``Foo`` and other things.
In this case, the application should link to ``FooPlugin`` directly,
before ``Foo``.  However, the application author only knows about ``Foo``.
We can express this as follows:

.. code-block:: cmake

  # Core library used by other components.
  add_library(Core STATIC core.cpp)

  # Foo is a static library for use by applications.
  # Implementation of Foo depends on Core.
  add_library(Foo STATIC foo.cpp foo_plugin_helper.cpp)
  target_link_libraries(Foo PRIVATE Core)

  # Extra parts of Foo for use by its static plugins.
  # Implementation of Foo's extra parts depends on both Core and Foo.
  add_library(FooExtras STATIC foo_extras.cpp)
  target_link_libraries(FooExtras PRIVATE Core Foo)

  # The Foo library has an associated static plugin
  # that should be linked into the final executable.
  # Implementation of the plugin depends on Core, Foo, and FooExtras.
  add_library(FooPlugin STATIC foo_plugin.cpp)
  target_link_libraries(FooPlugin PRIVATE Core Foo FooExtras)

  # An app that links Foo should link Foo's plugin directly.
  set_property(TARGET Foo PROPERTY INTERFACE_LINK_LIBRARIES_DIRECT FooPlugin)

  # An app does not need to link Foo directly because the plugin links it.
  set_property(TARGET Foo PROPERTY INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE Foo)

An application ``app`` only needs to specify that it links to ``Foo``:

.. code-block:: cmake

  add_executable(app main.cpp)
  target_link_libraries(app PRIVATE Foo)

The ``INTERFACE_LINK_LIBRARIES_DIRECT`` target property on ``Foo`` tells
CMake to pretend that ``app`` also links directly to ``FooPlugin``.
The ``INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE`` target property on ``Foo``
tells CMake to pretend that ``app`` did *not* link directly to ``Foo``.
Instead, ``Foo`` will be linked as a dependency of ``FooPlugin``.  The
final link line for ``app`` will link the libraries in the following
order:

* ``FooPlugin`` as a direct link dependency of ``app``
  (via ``Foo``'s usage requirements).
* ``FooExtras`` as a dependency of ``FooPlugin``.
* ``Foo`` as a dependency of ``FooPlugin`` and ``FooExtras``.
* ``Core`` as a dependency of ``FooPlugin``, ``FooExtras``, and ``Foo``.

Note that without the ``INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE`` target
property, ``Foo`` would be linked twice: once as a direct dependency
of ``app``, and once as a dependency of ``FooPlugin``.

Example: Opt-In Static Plugins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the above `Example: Static Plugins`_, the ``app`` executable specifies
that it links directly to ``Foo``.  In a real application, there might
be an intermediate library:

.. code-block:: cmake

  add_library(app_impl STATIC app_impl.cpp)
  target_link_libraries(app_impl PRIVATE Foo)

  add_executable(app main.cpp)
  target_link_libraries(app PRIVATE app_impl)

In this case we do not want ``Foo``'s ``INTERFACE_LINK_LIBRARIES_DIRECT``
and ``INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE`` target properties to affect
the direct dependencies of ``app_impl``.  To avoid this, we can revise
the property values to make their effects opt-in:

.. code-block:: cmake

  # An app that links Foo should link Foo's plugin directly.
  set_property(TARGET Foo PROPERTY INTERFACE_LINK_LIBRARIES_DIRECT
    "$<$<BOOL:$<TARGET_PROPERTY:FOO_STATIC_PLUGINS>>:FooPlugin>"
  )

  # An app does not need to link Foo directly because the plugin links it.
  set_property(TARGET Foo PROPERTY INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE
    "$<$<BOOL:$<TARGET_PROPERTY:FOO_STATIC_PLUGINS>>:Foo>"
  )

Now, the ``app`` executable can opt-in to get ``Foo``'s plugin(s):

.. code-block:: cmake

  set_property(TARGET app PROPERTY FOO_STATIC_PLUGINS 1)

The final link line for ``app`` will now link the libraries in the following
order:

* ``FooPlugin`` as a direct link dependency of ``app``
  (via ``Foo``'s usage requirements).
* ``app_impl`` as a direct link dependency of ``app``.
* ``FooExtras`` as a dependency of ``FooPlugin``.
* ``Foo`` as a dependency of ``app_impl``, ``FooPlugin``, and ``FooExtras``.
* ``Core`` as a dependency of ``FooPlugin``, ``FooExtras``, and ``Foo``.
