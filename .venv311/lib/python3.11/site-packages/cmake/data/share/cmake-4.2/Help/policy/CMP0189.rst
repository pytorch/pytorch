CMP0189
-------

.. versionadded:: 4.1

:genex:`TARGET_PROPERTY` evaluates ``LINK_LIBRARIES`` properties transitively.

The :prop_tgt:`LINK_LIBRARIES` and :prop_tgt:`INTERFACE_LINK_LIBRARIES`
target properties record link dependencies through which the
:genex:`TARGET_PROPERTY` generator expression evaluates transitive properties.
However, in CMake 4.0 and below, the properties themselves were not evaluated
transitively.  CMake 4.1 and above prefer to evaluate the
:prop_tgt:`LINK_LIBRARIES` and :prop_tgt:`INTERFACE_LINK_LIBRARIES`
target properties transitively because they are among the
:ref:`build specification <Target Build Specification>` and
:ref:`usage requirement <Target Usage Requirements>` properties, respectively.

This policy provides compatibility for projects that have not been updated to
expect the new behavior.  It takes effect during buildsystem generation.
Generator expressions are evaluated in each directory using the policy setting
as of the end of its ``CMakeLists.txt``.  As an exception, generator
expressions evaluated by the :command:`file(GENERATE)` command use the
policy setting as of its call site.

The ``OLD`` behavior of this policy is for :genex:`TARGET_PROPERTY` to not
evaluate :prop_tgt:`LINK_LIBRARIES` and :prop_tgt:`INTERFACE_LINK_LIBRARIES`
transitively.  The ``NEW`` behavior is for :genex:`TARGET_PROPERTY` to
evaluate :prop_tgt:`LINK_LIBRARIES` and :prop_tgt:`INTERFACE_LINK_LIBRARIES`
transitively.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 4.1
.. |WARNS_OR_DOES_NOT_WARN| replace:: does *not* warn
.. include:: include/STANDARD_ADVICE.rst

.. include:: include/DEPRECATED.rst
