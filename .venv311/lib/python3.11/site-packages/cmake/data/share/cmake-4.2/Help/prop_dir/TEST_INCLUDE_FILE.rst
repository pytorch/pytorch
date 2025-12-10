TEST_INCLUDE_FILE
-----------------

.. deprecated:: 3.10

  Use the :prop_dir:`TEST_INCLUDE_FILES` directory property instead, which
  supports specifying multiple files.

The ``TEST_INCLUDE_FILE`` directory property specifies a CMake script that is
included and processed when ``ctest`` is run on the directory.

If both the ``TEST_INCLUDE_FILE`` and :prop_dir:`TEST_INCLUDE_FILES` directory
properties are set, the script specified in ``TEST_INCLUDE_FILE`` is included
first, followed by the scripts listed in ``TEST_INCLUDE_FILES``.
