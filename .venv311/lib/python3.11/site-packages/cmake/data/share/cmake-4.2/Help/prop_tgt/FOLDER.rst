FOLDER
------

For IDEs that present targets using a folder hierarchy, this property
specifies the name of the folder to place the target under.
To nest folders, use ``FOLDER`` values such as ``GUI/Dialogs`` with ``/``
characters separating folder levels.  Targets with no ``FOLDER`` property
will appear as top level entities.  Targets with the same ``FOLDER``
property value will appear in the same folder as siblings.

Only some CMake generators honor the ``FOLDER`` property
(e.g. :generator:`Xcode` or any of the
:ref:`Visual Studio <Visual Studio Generators>` generators).
Those generators that don't will simply ignore it.

This property is initialized by the value of the variable
:variable:`CMAKE_FOLDER` if it is set when a target is created.

The global property :prop_gbl:`USE_FOLDERS` must be set to true, otherwise
the ``FOLDER`` property is ignored.
