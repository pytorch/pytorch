output_required_files
---------------------

Disallowed since version 3.0.  See CMake Policy :policy:`CMP0032`.

Approximate C preprocessor dependency scanning.

This command exists only because ancient CMake versions provided it.
CMake handles preprocessor dependency scanning automatically using a
more advanced scanner.

.. code-block:: cmake

  output_required_files(srcfile outputfile)

Outputs a list of all the source files that are required by the
specified ``srcfile``.  This list is written into ``outputfile``.  This is
similar to writing out the dependencies for ``srcfile`` except that it
jumps from ``.h`` files into ``.cxx``, ``.c`` and ``.cpp`` files if possible.
