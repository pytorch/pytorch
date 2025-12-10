CMAKE_FIND_PACKAGE_SORT_DIRECTION
---------------------------------

.. versionadded:: 3.7

.. versionchanged:: 4.2

  The default sort direction has changed from ``DEC`` to ``ASC``.


The sorting direction used by :variable:`CMAKE_FIND_PACKAGE_SORT_ORDER`.
It can assume one of the following values:

``ASC``
  Ordering is done in ascending mode.
  The lowest folder found will be tested first.

``DEC``
  Default. Ordering is done in descending mode.
  The highest folder found will be tested first.

If :variable:`CMAKE_FIND_PACKAGE_SORT_ORDER` is set to ``NONE`` this variable
has no effect.
