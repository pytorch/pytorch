.. note::

  If the compiler's default standard level is at least that
  of the requested feature, CMake may omit the ``-std=`` flag.
  The flag may still be added if the compiler's default extensions mode
  does not match the :prop_tgt:`<LANG>_EXTENSIONS` target property,
  or if the :prop_tgt:`<LANG>_STANDARD` target property is set.
