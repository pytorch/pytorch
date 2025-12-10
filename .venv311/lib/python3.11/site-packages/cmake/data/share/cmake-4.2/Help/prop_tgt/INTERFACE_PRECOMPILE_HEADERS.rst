INTERFACE_PRECOMPILE_HEADERS
----------------------------

.. versionadded:: 3.16

List of interface header files to precompile into consuming targets.

Targets may populate this property to publish the header files
for consuming targets to precompile.  The :command:`target_precompile_headers`
command populates this property with values given to the ``PUBLIC`` and
``INTERFACE`` keywords.  Projects may also get and set the property directly.
See the discussion in :command:`target_precompile_headers` for guidance on
appropriate use of this property for installed or exported targets.

Contents of ``INTERFACE_PRECOMPILE_HEADERS`` may use "generator expressions"
with the syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)`
manual for available expressions.  See the :manual:`cmake-buildsystem(7)`
manual for more on defining buildsystem properties.
