.. note::

  On macOS, this property will be ignored for the linker import files (e.g.
  ``.tbd`` files, see :prop_tgt:`ENABLE_EXPORTS` property for details) when:

  * The :prop_tgt:`FRAMEWORK` is set, because the framework layout cannot be
    changed.
  * The :generator:`Xcode` generator is used, due to the limitations and
    constraints of the ``Xcode`` tool.

  In both cases, the linker import files will be generated |IDEM| as the shared
  library.
