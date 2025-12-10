LINK_DEPENDS_NO_SHARED
----------------------

Do not depend on linked shared library files.

Set this property to true to tell CMake generators not to add
file-level dependencies on the shared library files linked by this
target.  Modification to the shared libraries will not be sufficient
to re-link this target.  Logical target-level dependencies will not be
affected so the linked shared libraries will still be brought up to
date before this target is built.

This property is initialized by the value of the
:variable:`CMAKE_LINK_DEPENDS_NO_SHARED` variable if it is set when a
target is created.
