CMAKE_CXX_COMPILER_IMPORT_STD
-----------------------------

.. versionadded:: 3.30

A list of C++ standard levels for which ``import std`` support exists for the
current C++ toolchain.  Support for C++\<NN\> may be detected using a
``<NN> IN_LIST CMAKE_CXX_COMPILER_IMPORT_STD`` predicate with the
:command:`if` command.

.. note::

   This variable is meaningful only when experimental support for ``import
   std;`` has been enabled by the ``CMAKE_EXPERIMENTAL_CXX_IMPORT_STD`` gate.
