LINK_LIBRARY_OVERRIDE_<LIBRARY>
-------------------------------

.. versionadded:: 3.24

Override the library feature associated with ``<LIBRARY>`` from
:genex:`LINK_LIBRARY` generator expressions.  This can be used to resolve
incompatible library features that result from specifying different features
for ``<LIBRARY>`` in different :genex:`LINK_LIBRARY` generator expressions.

When set on a target, this property holds a single library feature name, which
will be applied to ``<LIBRARY>`` when linking that target.

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
  set_property(TARGET lib3 PROPERTY LINK_LIBRARY_OVERRIDE_lib1 feature2)

  # lib1 will now be associated with feature2 instead when linking lib3

It is also possible to override any feature with the pre-defined ``DEFAULT``
library feature.  This effectively discards any feature for that link item,
for that target only (``lib3`` in this example):

.. code-block:: cmake

  # When linking lib3, discard any library feature for lib1
  set_property(TARGET lib3 PROPERTY LINK_LIBRARY_OVERRIDE_lib1 DEFAULT)

See the :prop_tgt:`LINK_LIBRARY_OVERRIDE` target property for an alternative
way of overriding library features for multiple libraries at once.  If both
properties are defined and specify an override for the same link item,
``LINK_LIBRARY_OVERRIDE_<LIBRARY>`` takes precedence over
:prop_tgt:`LINK_LIBRARY_OVERRIDE`.

Contents of ``LINK_LIBRARY_OVERRIDE_<LIBRARY>`` may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.

For more information about library features, see the
:variable:`CMAKE_<LANG>_LINK_LIBRARY_USING_<FEATURE>` and
:variable:`CMAKE_LINK_LIBRARY_USING_<FEATURE>` variables.
