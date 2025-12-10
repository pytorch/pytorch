CMAKE_COMPILER_IS_GNUG77
------------------------

.. deprecated:: 3.24

  Use the :variable:`CMAKE_Fortran_COMPILER_ID <CMAKE_<LANG>_COMPILER_ID>`
  variable instead.

  The ``CMAKE_COMPILER_IS_*`` variables were used in early CMake versions before
  the introduction of :variable:`CMAKE_<LANG>_COMPILER_ID` variables in CMake
  2.6.

The ``CMAKE_COMPILER_IS_GNUG77`` variable is set to boolean true if the
``Fortran`` compiler is GNU.

Examples
^^^^^^^^

In earlier versions of CMake, the ``CMAKE_COMPILER_IS_GNUG77`` variable was used
to check if the ``Fortran`` compiler was GNU:

.. code-block:: cmake

  if(CMAKE_COMPILER_IS_GNUG77)
    # GNU Fortran compiler-specific logic.
  endif()

Starting with CMake 2.6, the ``CMAKE_Fortran_COMPILER_ID`` variable should be
used instead:

.. code-block:: cmake

  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    # GNU Fortran compiler-specific logic.
  endif()
