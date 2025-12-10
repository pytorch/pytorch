CMAKE_FRAMEWORK_MULTI_CONFIG_POSTFIX_<CONFIG>
---------------------------------------------

.. versionadded:: 3.18

Default framework filename postfix under configuration ``<CONFIG>`` when
using a multi-config generator.

When a framework target is created its :prop_tgt:`FRAMEWORK_MULTI_CONFIG_POSTFIX_<CONFIG>`
target property is initialized with the value of this variable if it is set.
