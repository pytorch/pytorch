CMAKE_DEPENDS_IN_PROJECT_ONLY
-----------------------------

.. versionadded:: 3.6

When set to ``TRUE`` in a directory, the build system produced by the
:ref:`Makefile Generators` is set up to only consider dependencies on source
files that appear either in the source or in the binary directories.  Changes
to source files outside of these directories will not cause rebuilds.

This should be used carefully in cases where some source files are picked up
through external headers during the build.
