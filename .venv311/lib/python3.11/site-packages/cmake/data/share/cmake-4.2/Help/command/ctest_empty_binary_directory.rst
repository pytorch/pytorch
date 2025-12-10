ctest_empty_binary_directory
----------------------------

empties the binary directory

.. code-block:: cmake

  ctest_empty_binary_directory(<directory>)

Removes a binary directory.  This command will perform some checks
prior to deleting the directory in an attempt to avoid malicious or
accidental directory deletion.
