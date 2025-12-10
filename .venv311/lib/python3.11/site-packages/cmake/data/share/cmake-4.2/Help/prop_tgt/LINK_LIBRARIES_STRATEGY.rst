LINK_LIBRARIES_STRATEGY
-----------------------

.. versionadded:: 3.31

Specify a strategy for ordering a target's direct link dependencies
on linker command lines.  This property is initialized by the value of the
:variable:`CMAKE_LINK_LIBRARIES_STRATEGY` variable if it is set when a
target is created.

CMake generates a target's link line using its :ref:`Target Link Properties`.
In particular, the :prop_tgt:`LINK_LIBRARIES` target property records the
target's direct link dependencies, typically populated by calls to
:command:`target_link_libraries`.  Indirect link dependencies are
propagated from those entries of :prop_tgt:`LINK_LIBRARIES` that name
library targets by following the transitive closure of their
:prop_tgt:`INTERFACE_LINK_LIBRARIES` properties.  CMake supports multiple
strategies for nominally ordering direct and indirect link dependencies,
which are then filtered for `Toolchain-Specific Behavior`_.

Consider this example for the strategies below:

.. code-block:: cmake

  add_library(A STATIC ...)
  add_library(B STATIC ...)
  add_library(C STATIC ...)
  add_executable(main ...)
  target_link_libraries(B PRIVATE A)
  target_link_libraries(C PRIVATE A)
  target_link_libraries(main PRIVATE A B C)

The supported strategies are:

``REORDER_MINIMALLY``
  Entries of :prop_tgt:`LINK_LIBRARIES` always appear first and in their
  original order.  Indirect link dependencies not satisfied by the
  original entries may be reordered and de-duplicated with respect to
  one another, but are always appended after the original entries.
  This may result in less efficient link lines, but gives projects
  control of ordering among independent entries.  Such control may be
  important when intermixing link flags with libraries, or when multiple
  libraries provide a given symbol.

  This is the default.

  In the above example, this strategy computes a link line for ``main``
  by starting with its original entries ``A B C``, and then appends
  another ``A`` to satisfy the dependencies of ``B`` and ``C`` on ``A``.
  The nominal order produced by this strategy is ``A B C A``.

  Note that additional filtering for `Toolchain-Specific Behavior`_
  may de-duplicate ``A`` on the actual linker invocation in the
  generated build system, resulting in either ``A B C`` or ``B C A``.

``REORDER_FREELY``
  Entries of :prop_tgt:`LINK_LIBRARIES` may be reordered, de-duplicated,
  and intermixed with indirect link dependencies.  This may result in
  more efficient link lines, but does not give projects any control of
  ordering among independent entries.

  In the above example, this strategy computes a link line for ``main``
  by re-ordering its original entries ``A B C`` to satisfy the
  dependencies of ``B`` and ``C`` on ``A``.
  The nominal order produced by this strategy is ``B C A``.

Toolchain-Specific Behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^

After one of the above strategies produces a nominal order among
direct and indirect link dependencies, the actual linker invocation
in the generated build system may de-duplicate entries based on
platform-specific requirements and linker capabilities.
See policy :policy:`CMP0156`.

For example, if the ``REORDER_MINIMALLY`` strategy produces ``A B C A``,
the actual link line may de-duplicate ``A`` as follows:

* If ``A`` is a static library and the linker re-scans automatically,
  the first occurrence is kept, resulting in ``A B C``.
  See policy :policy:`CMP0179`

* If ``A`` is a shared library on Windows, the first
  occurrence is kept, resulting in ``A B C``.

* If ``A`` is a shared library on macOS or UNIX platforms, the last
  occurrence is kept, resulting in ``B C A``.
