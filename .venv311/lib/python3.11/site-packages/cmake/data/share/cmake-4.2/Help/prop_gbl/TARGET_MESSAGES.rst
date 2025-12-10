TARGET_MESSAGES
---------------

.. versionadded:: 3.4

Specify whether to report the completion of each target.

This property specifies whether :ref:`Makefile Generators` should
add a progress message describing that each target has been completed.
If the property is not set the default is ``ON``.  Set the property
to ``OFF`` to disable target completion messages.

This option is intended to reduce build output when little or no
work needs to be done to bring the build tree up to date.

If a ``CMAKE_TARGET_MESSAGES`` cache entry exists its value
initializes the value of this property.

Non-Makefile generators currently ignore this property.

See the counterpart property :prop_gbl:`RULE_MESSAGES` to disable
everything except for target completion messages.
