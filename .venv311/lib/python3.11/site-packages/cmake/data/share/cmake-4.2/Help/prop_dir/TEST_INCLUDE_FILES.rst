TEST_INCLUDE_FILES
------------------

.. versionadded:: 3.10

This directory property specifies a list of CMake scripts to be included and
processed when ``ctest`` runs on the directory.  Use absolute paths, to avoid
ambiguity.  Script files are included in the specified order.

``TEST_INCLUDE_FILES`` scripts are processed when running ``ctest``, not during
the ``cmake`` configuration phase.  These scripts should be written as if they
were CTest dashboard scripts.  It is common to generate such scripts dynamically
since many variables and commands available during configuration are not
accessible at test phase.

Examples
^^^^^^^^

Setting this directory property to append one or more CMake scripts:

.. code-block:: cmake
  :caption: CMakeLists.txt

  configure_file(script.cmake.in script.cmake)

  set_property(
    DIRECTORY
    APPEND
    PROPERTY TEST_INCLUDE_FILES
      ${CMAKE_CURRENT_BINARY_DIR}/script.cmake
      ${CMAKE_CURRENT_SOURCE_DIR}/foo.cmake
      ${dir}/bar.cmake
  )

.. code-block:: cmake
  :caption: script.cmake.in

  execute_process(
    COMMAND "@CMAKE_COMMAND@" -E echo "script.cmake executed during CTest"
  )
