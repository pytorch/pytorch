LINKER_TYPE
-----------

.. versionadded:: 3.29

Specify which linker will be used for the link step. The property value may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.

.. include:: ../variable/include/LINKER_PREDEFINED_TYPES.rst

This property is not supported on :generator:`Green Hills MULTI` generator.

The implementation details for the selected linker will be provided by the
:variable:`CMAKE_<LANG>_USING_LINKER_<TYPE>` variable. For example:

.. code-block:: cmake

  add_library(lib1 SHARED ...)
  set_property(TARGET lib1 PROPERTY LINKER_TYPE LLD)

This specifies that ``lib1`` should use linker type ``LLD`` for the link step.
The command line options that will be passed to the toolchain will be provided
by the ``CMAKE_<LANG>_USING_LINKER_LLD`` variable.

Note that the linker would typically be set using :variable:`CMAKE_LINKER_TYPE`
for the whole build rather than setting the ``LINKER_TYPE`` property on
individual targets.
