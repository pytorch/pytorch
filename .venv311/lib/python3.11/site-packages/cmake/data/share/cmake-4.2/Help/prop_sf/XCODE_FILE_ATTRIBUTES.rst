XCODE_FILE_ATTRIBUTES
---------------------

.. versionadded:: 3.7

Add values to the :generator:`Xcode` ``ATTRIBUTES`` setting on its reference to a
source file.  Among other things, this can be used to set the role on
a ``.mig`` file:

.. code-block:: cmake

  set_source_files_properties(defs.mig
      PROPERTIES
          XCODE_FILE_ATTRIBUTES "Client;Server"
  )
