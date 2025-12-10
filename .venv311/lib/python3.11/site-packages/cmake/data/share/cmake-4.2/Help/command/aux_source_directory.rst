aux_source_directory
--------------------

Find all source files in a directory.

.. code-block:: cmake

  aux_source_directory(<dir> <variable>)

Collects the names of all the source files in the specified directory
and stores the list in the ``<variable>`` provided.  This command is
intended to be used by projects that use explicit template
instantiation.  Template instantiation files can be stored in a
``Templates`` subdirectory and collected automatically using this
command to avoid manually listing all instantiations.

It is tempting to use this command to avoid writing the list of source
files for a library or executable target.  While this seems to work,
there is no way for CMake to generate a build system that knows when a
new source file has been added.  Normally the generated build system
knows when it needs to rerun CMake because the ``CMakeLists.txt`` file is
modified to add a new source.  When the source is just added to the
directory without modifying this file, one would have to manually
rerun CMake to generate a build system incorporating the new file.
