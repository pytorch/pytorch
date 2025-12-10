make_directory
--------------

.. deprecated:: 3.0

  Use the :command:`file(MAKE_DIRECTORY)` command instead.

.. code-block:: cmake

  make_directory(directory)

Creates the specified directory.  Full paths should be given.  Any
parent directories that do not exist will also be created.  Use with
care.
