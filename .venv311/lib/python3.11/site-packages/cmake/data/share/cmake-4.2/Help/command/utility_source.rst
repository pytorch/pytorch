utility_source
--------------

Disallowed since version 3.0.  See CMake Policy :policy:`CMP0034`.

Specify the source tree of a third-party utility.

.. code-block:: cmake

  utility_source(cache_entry executable_name
                 path_to_source [file1 file2 ...])

When a third-party utility's source is included in the distribution,
this command specifies its location and name.  The cache entry will
not be set unless the ``path_to_source`` and all listed files exist.  It
is assumed that the source tree of the utility will have been built
before it is needed.

When cross compiling CMake will print a warning if a ``utility_source()``
command is executed, because in many cases it is used to build an
executable which is executed later on.  This doesn't work when cross
compiling, since the executable can run only on their target platform.
So in this case the cache entry has to be adjusted manually so it
points to an executable which is runnable on the build host.
