CMAKE_<LANG>_USING_LINKER_MODE
------------------------------

.. deprecated:: 4.0

  This variable is no longer used.  The type of information stored in the
  :variable:`CMAKE_<LANG>_USING_LINKER_<TYPE>` variable is determined by
  the :variable:`CMAKE_<LANG>_LINK_MODE` variable.

.. versionadded:: 3.29

This variable controls how the value of the
:variable:`CMAKE_<LANG>_USING_LINKER_<TYPE>` variable should be interpreted.
The supported linker mode values are:

``FLAG``
  :variable:`CMAKE_<LANG>_USING_LINKER_<TYPE>` holds a
  :ref:`semicolon-separated list <CMake Language Lists>` of flags to be passed
  to the compiler frontend.  This is also the default behavior if
  ``CMAKE_<LANG>_USING_LINKER_MODE`` is not set.

``TOOL``
  :variable:`CMAKE_<LANG>_USING_LINKER_<TYPE>` holds the path to the linker
  tool.

.. warning::

  The variable must be set accordingly to how CMake manage the link step:

  * value ``TOOL`` is expected and required when the linker is used directly
    for the link step.
  * value ``FLAG`` is expected or the variable not set when the compiler is
    used as driver for the link step.
