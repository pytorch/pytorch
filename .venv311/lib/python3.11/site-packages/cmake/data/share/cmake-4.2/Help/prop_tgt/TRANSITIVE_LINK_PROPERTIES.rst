TRANSITIVE_LINK_PROPERTIES
--------------------------

.. versionadded:: 3.30

Properties that the :genex:`TARGET_PROPERTY` generator expression, on the
target and its dependents, evaluates as the union of values collected from
the transitive closure of link dependencies, including entries guarded by
:genex:`LINK_ONLY`.

The value is a :ref:`semicolon-separated list <CMake Language Lists>`
of :ref:`custom transitive property <Custom Transitive Properties>` names.
Any leading ``INTERFACE_`` prefix is ignored, e.g., ``INTERFACE_PROP`` is
treated as just ``PROP``.

See documentation of the :genex:`TARGET_PROPERTY` generator expression
for details of custom transitive property evaluation.  See also the
:prop_tgt:`TRANSITIVE_COMPILE_PROPERTIES` target property, which excludes
entries guarded by :genex:`LINK_ONLY`..
