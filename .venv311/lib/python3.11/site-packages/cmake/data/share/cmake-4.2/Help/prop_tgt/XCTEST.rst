XCTEST
------

.. versionadded:: 3.3

Boolean target property that indicates whether a target is an XCTest CFBundle
(Core Foundation Bundle) on Apple systems.

This property is usually set automatically by the :command:`xctest_add_bundle`
command provided by the :module:`FindXCTest` module.

If a module library target has this property set to boolean true, it will be
built as a CFBundle when built on Apple system, with the required CFBundle
directory structure.

This property depends on :prop_tgt:`BUNDLE` target property to be effective.
