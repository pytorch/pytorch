PUBLIC_HEADER
-------------

Specify public header files in a :prop_tgt:`FRAMEWORK` shared library target.

Shared library targets marked with the :prop_tgt:`FRAMEWORK` property generate
frameworks on macOS, iOS and normal shared libraries on other platforms.
This property may be set to a list of header files to be placed in the
``Headers`` directory inside the framework folder.  On non-Apple platforms
these headers may be installed using the ``PUBLIC_HEADER`` option to the
:command:`install(TARGETS)` command.
