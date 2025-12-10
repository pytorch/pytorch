LINK_LIBRARY_OVERRIDE
---------------------

.. versionadded:: 3.24

Override the library features associated with libraries from
:genex:`LINK_LIBRARY` generator expressions.  This can be used to resolve
incompatible library features that result from specifying different features
for the same library in different :genex:`LINK_LIBRARY` generator expressions.

This property supports overriding multiple libraries and features. It expects
a :ref:`semicolon-separated list <CMake Language Lists>`, where each list item
has the following form::

  feature[,link-item]*

For each comma-separated ``link-item``, any existing library feature associated
with it will be ignored for the target this property is set on.  The item
will instead be associated with the specified ``feature``.  Each ``link-item``
can be anything that would be accepted as part of a ``library-list`` in a
:genex:`LINK_LIBRARY` generator expression.

.. code-block:: cmake

  add_library(lib1 ...)
  add_library(lib2 ...)
  add_library(lib3 ...)

  target_link_libraries(lib1 PUBLIC "$<LINK_LIBRARY:feature1,external>")
  target_link_libraries(lib2 PUBLIC "$<LINK_LIBRARY:feature2,lib1>")
  target_link_libraries(lib3 PRIVATE lib1 lib2)

  # lib1 is associated with both feature2 and no feature. Without any override,
  # this would result in a fatal error at generation time for lib3.
  # Define an override to resolve the incompatible feature associations.
  set_property(TARGET lib3 PROPERTY LINK_LIBRARY_OVERRIDE "feature2,lib1,external")

  # lib1 and external will now be associated with feature2 instead when linking lib3

It is also possible to override any feature with the pre-defined ``DEFAULT``
library feature.  This effectively discards any feature for that link item,
for that target only (``lib3`` in this example):

.. code-block:: cmake

  # When linking lib3, discard any library feature for lib1, and use feature2 for external
  set_property(TARGET lib3 PROPERTY LINK_LIBRARY_OVERRIDE
    "DEFAULT,lib1"
    "feature2,external"
  )

The above example also demonstrates how to specify different feature overrides
for different link items.  See the :prop_tgt:`LINK_LIBRARY_OVERRIDE_<LIBRARY>`
target property for an alternative way of overriding library features for
individual libraries, which may be simpler in some cases.  If both properties
are defined and specify an override for the same link item,
:prop_tgt:`LINK_LIBRARY_OVERRIDE_<LIBRARY>` takes precedence over
``LINK_LIBRARY_OVERRIDE``.

Contents of ``LINK_LIBRARY_OVERRIDE`` may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.

For more information about library features, see the
:variable:`CMAKE_<LANG>_LINK_LIBRARY_USING_<FEATURE>` and
:variable:`CMAKE_LINK_LIBRARY_USING_<FEATURE>` variables.
