HEADER_FILE_ONLY
----------------

Is this source file only a header file.

A property on a source file that indicates if the source file is a
header file with no associated implementation.  This is set
automatically based on the file extension and is used by CMake to
determine if certain dependency information should be computed.

By setting this property to ``ON``, you can disable compilation of
the given source file, even if it should be compiled because it is
part of the library's/executable's sources.

This is useful if you have some source files which you somehow
pre-process, and then add these pre-processed sources via
:command:`add_library` or :command:`add_executable`. Normally, in IDE,
there would be no reference of the original sources, only of these
pre-processed sources. So by setting this property for all the original
source files to ``ON``, and then either calling :command:`add_library`
or :command:`add_executable` while passing both the pre-processed
sources and the original sources, or by using :command:`target_sources`
to add original source files will do exactly what would one expect, i.e.
the original source files would be visible in IDE, and will not be built.
