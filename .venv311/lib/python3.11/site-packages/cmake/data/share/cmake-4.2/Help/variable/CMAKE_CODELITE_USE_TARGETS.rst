CMAKE_CODELITE_USE_TARGETS
--------------------------

.. versionadded:: 3.7

Change the way the CodeLite generator creates projectfiles.

If this variable evaluates to ``ON`` at the end of the top-level
``CMakeLists.txt`` file, the generator creates projectfiles based on targets
rather than projects.
