INSTALL_OBJECT_NAME
-------------------

.. versionadded:: 4.2

Set the installed object name (without the object extension) of the source
file. An empty string value disables custom object naming. The value must be a
relative path, and may not include special directory components (e.g.,
``..``).

Note that the object name might not be used as-is in some
:prop_tgt:`INSTALL_OBJECT_NAME_STRATEGY` strategies.  It may be changed as
the strategy requires to fulfill its goals.

This property supports
:manual:`generator expressions <cmake-generator-expressions(7)>`.

.. note::
   No collision resistance within a target is performed by CMake. When using
   this property, collisions must be avoided in the project code. CMake has a
   number of source files it generates that also create object files that may
   collide with a given custom name. These include:

   * Generated PCH source files (``cmake_pch``)
   * Generated Unity compilation files (``unity_...``)
   * Qt autogen sources (``moc_compilations.cpp``)
