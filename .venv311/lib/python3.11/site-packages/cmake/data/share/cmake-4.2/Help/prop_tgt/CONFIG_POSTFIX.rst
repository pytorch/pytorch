<CONFIG>_POSTFIX
----------------

Postfix to append to the target file name for configuration ``<CONFIG>``.

When building with configuration ``<CONFIG>`` the value of this property
is appended to the target file name built on disk.  For non-executable
targets, this property is initialized by the value of the
:variable:`CMAKE_<CONFIG>_POSTFIX` variable if it is set when a target is
created.  This property is ignored on macOS for Frameworks and App Bundles.

For macOS see also the :prop_tgt:`FRAMEWORK_MULTI_CONFIG_POSTFIX_<CONFIG>`
target property.
