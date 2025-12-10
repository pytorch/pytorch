INTERFACE_HEADER_SETS_TO_VERIFY
-------------------------------

.. versionadded:: 3.24

Used to specify which ``PUBLIC`` and ``INTERFACE`` header sets of a target
should be verified.

This property contains a semicolon-separated list of header sets which
should be verified if :prop_tgt:`VERIFY_INTERFACE_HEADER_SETS` is set to
``TRUE``. If the list is empty, all ``PUBLIC`` and ``INTERFACE`` header sets
are verified. (If the project does not want to verify any header sets on the
target, simply set :prop_tgt:`VERIFY_INTERFACE_HEADER_SETS` to ``FALSE``.)
