CMP0063
-------

.. |REMOVED_IN_CMAKE_VERSION| replace:: 4.0
.. include:: include/REMOVED_PROLOGUE.rst

.. versionadded:: 3.3

Honor visibility properties for all target types.

The :prop_tgt:`<LANG>_VISIBILITY_PRESET` and
:prop_tgt:`VISIBILITY_INLINES_HIDDEN` target properties affect visibility
of symbols during dynamic linking.  When first introduced these properties
affected compilation of sources only in shared libraries, module libraries,
and executables with the :prop_tgt:`ENABLE_EXPORTS` property set.  This
was sufficient for the basic use cases of shared libraries and executables
with plugins.  However, some sources may be compiled as part of static
libraries or object libraries and then linked into a shared library later.
CMake 3.3 and above prefer to honor these properties for sources compiled
in all target types.  This policy preserves compatibility for projects
expecting the properties to work only for some target types.

The ``OLD`` behavior for this policy is to ignore the visibility properties
for static libraries, object libraries, and executables without exports.
The ``NEW`` behavior for this policy is to honor the visibility properties
for all target types.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 3.3
.. |WARNED_OR_DID_NOT_WARN| replace:: warned
.. include:: include/REMOVED_EPILOGUE.rst
