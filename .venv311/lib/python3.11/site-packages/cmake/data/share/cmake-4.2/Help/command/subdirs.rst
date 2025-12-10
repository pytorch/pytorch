subdirs
-------

.. deprecated:: 3.0

  Use the :command:`add_subdirectory` command instead.

Add a list of subdirectories to the build.

.. code-block:: cmake

  subdirs(dir1 dir2 ...[EXCLUDE_FROM_ALL exclude_dir1 exclude_dir2 ...]
          [PREORDER])

Add a list of subdirectories to the build.  The :command:`add_subdirectory`
command should be used instead of ``subdirs`` although ``subdirs`` will still
work.  This will cause any CMakeLists.txt files in the sub directories
to be processed by CMake.  Any directories after the ``PREORDER`` flag are
traversed first by makefile builds, the ``PREORDER`` flag has no effect on
IDE projects.  Any directories after the ``EXCLUDE_FROM_ALL`` marker will
not be included in the top level makefile or project file.  This is
useful for having CMake create makefiles or projects for a set of
examples in a project.  You would want CMake to generate makefiles or
project files for all the examples at the same time, but you would not
want them to show up in the top level project or be built each time
make is run from the top.
