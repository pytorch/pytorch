CMAKE_SUBLIME_TEXT_2_EXCLUDE_BUILD_TREE
---------------------------------------

.. versionadded:: 3.8

If this variable evaluates to ``ON`` at the end of the top-level
``CMakeLists.txt`` file, the :generator:`Sublime Text 2` extra generator
excludes the build tree from the ``.sublime-project`` if it is inside the
source tree.
