CPACK_CUSTOM_INSTALL_VARIABLES
------------------------------

.. versionadded:: 3.21

CPack variables (set via e.g. :option:`cpack -D`, ``CPackConfig.cmake`` or
:variable:`CPACK_PROJECT_CONFIG_FILE` scripts) are not directly visible in
installation scripts.  Instead, one can pass a list of ``varName=value``
pairs in the ``CPACK_CUSTOM_INSTALL_VARIABLES`` variable.  At install time,
each list item will result in a variable of the specified name (``varName``)
being set to the given ``value``.  The ``=`` can be omitted for an empty
``value``.

``CPACK_CUSTOM_INSTALL_VARIABLES`` allows the packaging installation to be
influenced by the user or driving script at CPack runtime without having to
regenerate the install scripts.

Example
"""""""

.. code-block:: cmake

  install(FILES large.txt DESTINATION data)

  install(CODE [[
    if(ENABLE_COMPRESSION)
      # "run-compressor" is a fictional tool that produces
      # large.txt.xz from large.txt and then removes the input file
      execute_process(COMMAND run-compressor $ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/large.txt)
    endif()
  ]])

With the above example snippet, :manual:`cpack <cpack(1)>` will by default
run the installation script with ``ENABLE_COMPRESSION`` unset, resulting in
a package containing the uncompressed ``large.txt``.  This can be overridden
when invoking :manual:`cpack <cpack(1)>` like so:

.. code-block:: shell

  cpack -D "CPACK_CUSTOM_INSTALL_VARIABLES=ENABLE_COMPRESSION=TRUE"

The installation script will then run with ``ENABLE_COMPRESSION`` set to
``TRUE``, resulting in a package containing the compressed ``large.txt.xz``
instead.
