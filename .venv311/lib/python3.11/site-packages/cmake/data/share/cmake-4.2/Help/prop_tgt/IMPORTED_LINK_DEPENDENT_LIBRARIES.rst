IMPORTED_LINK_DEPENDENT_LIBRARIES
---------------------------------

Dependent shared libraries of an imported shared library.

Shared libraries may be linked to other shared libraries as part of
their implementation.  On some platforms the linker searches for the
dependent libraries of shared libraries they are including in the
link.  Set this property to the list of dependent shared libraries of
an imported library.  The list should be disjoint from the list of
interface libraries in the :prop_tgt:`INTERFACE_LINK_LIBRARIES` property.  On
platforms requiring dependent shared libraries to be found at link
time CMake uses this list to add appropriate files or paths to the
link command line.  Ignored for non-imported targets.
