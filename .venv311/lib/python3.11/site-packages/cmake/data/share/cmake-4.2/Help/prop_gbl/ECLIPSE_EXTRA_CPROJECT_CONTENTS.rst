ECLIPSE_EXTRA_CPROJECT_CONTENTS
-------------------------------

.. versionadded:: 3.12

Additional contents to be inserted into the generated Eclipse cproject file.

The cproject file defines the CDT specific information. Some third party IDE's
are based on Eclipse with the addition of other information specific to that IDE.
Through this property, it is possible to add this additional contents to
the generated project.
It is expected to contain valid XML.

Also see the :prop_gbl:`ECLIPSE_EXTRA_NATURES` property.
