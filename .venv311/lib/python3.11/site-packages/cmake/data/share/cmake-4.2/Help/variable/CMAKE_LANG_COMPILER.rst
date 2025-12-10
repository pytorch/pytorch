CMAKE_<LANG>_COMPILER
---------------------

The full path to the compiler for ``LANG``.

This is the command that will be used as the ``<LANG>`` compiler.  Once
set, you can not change this variable.

Usage
^^^^^

This variable can be set by the user during the first time a build tree is configured.

If a non-full path value is supplied then CMake will resolve the full path of
the compiler.

The variable could be set in a user supplied toolchain file or via
:option:`-D <cmake -D>` on the command line.

.. note::
  Options that are required to make the compiler work correctly can be included
  as items in a list; they can not be changed.

.. code-block:: cmake

  #set within user supplied toolchain file
  set(CMAKE_C_COMPILER /full/path/to/qcc --arg1 --arg2)

or

.. code-block:: console

  $ cmake ... -DCMAKE_C_COMPILER='qcc;--arg1;--arg2'
