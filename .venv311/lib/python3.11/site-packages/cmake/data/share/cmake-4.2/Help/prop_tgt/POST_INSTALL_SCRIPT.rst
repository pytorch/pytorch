POST_INSTALL_SCRIPT
-------------------

Deprecated install support.

The :prop_tgt:`PRE_INSTALL_SCRIPT` and ``POST_INSTALL_SCRIPT`` properties are
the old way to specify CMake scripts to run before and after installing a
target.  They are used only when the old ``INSTALL_TARGETS`` command is
used to install the target.  Use the :command:`install` command instead.
