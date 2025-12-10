CMAKE_FIND_PACKAGE_TARGETS_GLOBAL
---------------------------------

.. versionadded:: 3.24

Setting to ``TRUE`` promotes all :prop_tgt:`IMPORTED` targets discovered
by :command:`find_package` to a ``GLOBAL`` scope.


Setting this to ``TRUE`` is akin to specifying ``GLOBAL``
as an argument to :command:`find_package`.
Default value is ``OFF``.
