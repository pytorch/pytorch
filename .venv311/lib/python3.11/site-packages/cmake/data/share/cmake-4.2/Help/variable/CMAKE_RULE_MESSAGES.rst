CMAKE_RULE_MESSAGES
-------------------

.. versionadded:: 3.13

Specify whether to report a message for each make rule.

If set in the cache it is used to initialize the value of the :prop_gbl:`RULE_MESSAGES` property.
Users may disable the option in their local build tree to disable granular messages
and report only as each target completes in Makefile builds.
