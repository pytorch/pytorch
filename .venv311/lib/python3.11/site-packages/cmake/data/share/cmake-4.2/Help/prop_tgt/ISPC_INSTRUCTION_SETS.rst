ISPC_INSTRUCTION_SETS
---------------------

.. versionadded:: 3.19

List of instruction set architectures to generate code for.

This property is initialized by the value of the :variable:`CMAKE_ISPC_INSTRUCTION_SETS`
variable if it is set when a target is created.

The ``ISPC_INSTRUCTION_SETS`` target property must be used when generating for multiple
instruction sets so that CMake can track what object files will be generated.

Examples
^^^^^^^^

.. code-block:: cmake

  set_property(TARGET tgt PROPERTY ISPC_INSTRUCTION_SETS avx2-i32x4 avx512skx-i32x835)

Generates code for avx2 and avx512skx target architectures.
