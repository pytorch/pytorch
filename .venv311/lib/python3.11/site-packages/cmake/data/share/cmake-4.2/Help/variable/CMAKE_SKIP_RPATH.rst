CMAKE_SKIP_RPATH
----------------

If true, do not add run time path information.

If this is set to ``TRUE``, then the rpath information is not added to
compiled executables.  The default is to add rpath information if the
platform supports it.  This allows for easy running from the build
tree.  To omit RPATH in the install step, but not the build step, use
:variable:`CMAKE_SKIP_INSTALL_RPATH` instead. To omit RPATH in the build step,
use :variable:`CMAKE_SKIP_BUILD_RPATH`.

For more information on RPATH handling see the :prop_tgt:`INSTALL_RPATH`
and :prop_tgt:`BUILD_RPATH` target properties.
