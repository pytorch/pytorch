LISTFILE_STACK
--------------

The current stack of listfiles being processed.

This property is mainly useful when trying to debug errors in your
CMake scripts.  It returns a list of what list files are currently
being processed, in order.  So if one listfile does an
:command:`include` command then that is effectively pushing the
included listfile onto the stack.
