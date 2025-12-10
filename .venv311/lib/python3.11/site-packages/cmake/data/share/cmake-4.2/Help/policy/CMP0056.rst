CMP0056
-------

.. |REMOVED_IN_CMAKE_VERSION| replace:: 4.0
.. include:: include/REMOVED_PROLOGUE.rst

.. versionadded:: 3.2

Honor link flags in :command:`try_compile` source-file signature.

The :command:`try_compile` command source-file signature generates a
``CMakeLists.txt`` file to build the source file into an executable.
In order to compile the source the same way as it might be compiled
by the calling project, the generated project sets the value of the
:variable:`CMAKE_<LANG>_FLAGS` variable to that in the calling project.
The value of the :variable:`CMAKE_EXE_LINKER_FLAGS` variable may be
needed in some cases too, but CMake 3.1 and lower did not set it in
the generated project.  CMake 3.2 and above prefer to set it so that
linker flags are honored as well as compiler flags.  This policy
provides compatibility with the pre-3.2 behavior.

The ``OLD`` behavior for this policy is to not set the value of the
:variable:`CMAKE_EXE_LINKER_FLAGS` variable in the generated test
project.  The ``NEW`` behavior for this policy is to set the value of
the :variable:`CMAKE_EXE_LINKER_FLAGS` variable in the test project
to the same as it is in the calling project.

If the project code does not set the policy explicitly, users may
set it on the command line by defining the
:variable:`CMAKE_POLICY_DEFAULT_CMP0056 <CMAKE_POLICY_DEFAULT_CMP<NNNN>>`
variable in the cache.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 3.2
.. |WARNED_OR_DID_NOT_WARN| replace:: did *not* warn by default
.. include:: include/REMOVED_EPILOGUE.rst

See documentation of the
:variable:`CMAKE_POLICY_WARNING_CMP0056 <CMAKE_POLICY_WARNING_CMP<NNNN>>`
variable to control the warning in CMake versions before 4.0.
