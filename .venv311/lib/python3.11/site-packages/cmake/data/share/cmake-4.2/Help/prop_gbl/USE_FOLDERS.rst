USE_FOLDERS
-----------

Controls whether to use the :prop_tgt:`FOLDER` target property to organize
targets into folders.  The value of ``USE_FOLDERS`` at the end of the top level
``CMakeLists.txt`` file is what determines the behavior.

.. versionchanged:: 3.26

  CMake treats this property as ``ON`` by default.
  See policy :policy:`CMP0143`.

Not all CMake generators support recording folder details for targets.
The :generator:`Xcode` and :ref:`Visual Studio <Visual Studio Generators>`
generators are examples of generators that do.  Similarly, not all IDEs
support presenting targets using folder hierarchies, even if the CMake
generator used provides the necessary information.
