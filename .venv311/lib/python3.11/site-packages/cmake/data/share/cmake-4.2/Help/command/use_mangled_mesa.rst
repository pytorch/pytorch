use_mangled_mesa
----------------

Disallowed since version 3.0.  See CMake Policy :policy:`CMP0030`.

Copy mesa headers for use in combination with system GL.

.. code-block:: cmake

  use_mangled_mesa(PATH_TO_MESA OUTPUT_DIRECTORY)

The path to mesa includes, should contain ``gl_mangle.h``.  The mesa
headers are copied to the specified output directory.  This allows
mangled mesa headers to override other GL headers by being added to
the include directory path earlier.
