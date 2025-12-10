FRAMEWORK_MULTI_CONFIG_POSTFIX_<CONFIG>
---------------------------------------

.. versionadded:: 3.18

Postfix to append to the framework file name for configuration ``<CONFIG>``,
when using a multi-config generator (like :generator:`Xcode` and
:generator:`Ninja Multi-Config`).

When building with configuration ``<CONFIG>`` the value of this property
is appended to the framework file name built on disk.

For example, given a framework called ``my_fw``, a value of ``_debug``
for the ``FRAMEWORK_MULTI_CONFIG_POSTFIX_DEBUG`` property, and
``Debug;Release`` in :variable:`CMAKE_CONFIGURATION_TYPES`, the following
relevant files would be created for the ``Debug`` and ``Release``
configurations:

- ``Release/my_fw.framework/my_fw``
- ``Release/my_fw.framework/Versions/A/my_fw``
- ``Debug/my_fw.framework/my_fw_debug``
- ``Debug/my_fw.framework/Versions/A/my_fw_debug``

For framework targets, this property is initialized by the value of the
:variable:`CMAKE_FRAMEWORK_MULTI_CONFIG_POSTFIX_<CONFIG>` variable if it
is set when a target is created.

This property is ignored for non-framework targets, and when using single
config generators.
