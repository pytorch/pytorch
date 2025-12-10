CMP0051
-------

.. |REMOVED_IN_CMAKE_VERSION| replace:: 4.0
.. include:: include/REMOVED_PROLOGUE.rst

.. versionadded:: 3.1

List :genex:`TARGET_OBJECTS` in SOURCES target property.

CMake 3.0 and lower did not include the ``TARGET_OBJECTS``
:manual:`generator expression <cmake-generator-expressions(7)>` when
returning the :prop_tgt:`SOURCES` target property.

Configure-time CMake code is not able to handle generator expressions.  If
using the :prop_tgt:`SOURCES` target property at configure time, it may be
necessary to first remove generator expressions using the
:command:`string(GENEX_STRIP)` command.  Generate-time CMake code such as
:command:`file(GENERATE)` can handle the content without stripping.

The ``OLD`` behavior for this policy is to omit ``TARGET_OBJECTS``
expressions from the :prop_tgt:`SOURCES` target property.  The ``NEW``
behavior for this policy is to include ``TARGET_OBJECTS`` expressions
in the output.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 3.1
.. |WARNED_OR_DID_NOT_WARN| replace:: warned
.. include:: include/REMOVED_EPILOGUE.rst
