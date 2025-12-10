INTERFACE_INCLUDE_DIRECTORIES
-----------------------------

.. |property_name| replace:: include directories
.. |command_name| replace:: :command:`target_include_directories`
.. |PROPERTY_INTERFACE_NAME| replace:: ``INTERFACE_INCLUDE_DIRECTORIES``
.. |PROPERTY_LINK| replace:: :prop_tgt:`INCLUDE_DIRECTORIES`
.. include:: include/INTERFACE_BUILD_PROPERTY.rst

Include directories usage requirements commonly differ between the build-tree
and the install-tree.  The ``BUILD_INTERFACE`` and ``INSTALL_INTERFACE``
generator expressions can be used to describe separate usage requirements
based on the usage location.  Relative paths are allowed within the
``INSTALL_INTERFACE`` expression and are interpreted relative to the
installation prefix.  For example:

.. code-block:: cmake

  target_include_directories(mylib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/mylib>
    $<INSTALL_INTERFACE:include/mylib>  # <prefix>/include/mylib
  )

Creating Relocatable Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |INTERFACE_PROPERTY_LINK| replace:: ``INTERFACE_INCLUDE_DIRECTORIES``
.. include:: /include/INTERFACE_INCLUDE_DIRECTORIES_WARNING.rst
