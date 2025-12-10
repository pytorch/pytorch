CMAKE_MFC_FLAG
--------------

Use the MFC library for an executable or dll.

Enables the use of the Microsoft Foundation Classes (MFC).
It should be set to ``1`` for the static MFC library, and
``2`` for the shared MFC library.  This is used in Visual Studio
project files.

Contents of ``CMAKE_MFC_FLAG`` may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.

Examples
^^^^^^^^

Usage example:

.. code-block:: cmake

  set(CMAKE_MFC_FLAG 2)

  add_executable(CMakeSetup WIN32 ${SRCS})

  # Visual Studio generators add this flag automatically based on the
  # CMAKE_MFC_FLAG value, but generators matching "Make" require it:
  target_compile_definitions(CMakeSetup PRIVATE _AFXDLL)

See Also
^^^^^^^^

* The :module:`FindMFC` module to check whether MFC is installed and available.
