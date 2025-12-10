CMAKE_POLICY_WARNING_CMP<NNNN>
------------------------------

Explicitly enable or disable the warning when CMake Policy ``CMP<NNNN>``
has not been set explicitly by :command:`cmake_policy` or implicitly
by :command:`cmake_minimum_required`. This is meaningful
only for the policies that do not warn by default:

* ``CMAKE_POLICY_WARNING_CMP0025`` controlled the warning for
  policy :policy:`CMP0025` in CMake versions before 4.0.
* ``CMAKE_POLICY_WARNING_CMP0047`` controlled the warning for
  policy :policy:`CMP0047` in CMake versions before 4.0.
* ``CMAKE_POLICY_WARNING_CMP0056`` controlled the warning for
  policy :policy:`CMP0056` in CMake versions before 4.0.
* ``CMAKE_POLICY_WARNING_CMP0060`` controlled the warning for
  policy :policy:`CMP0060` in CMake versions before 4.0.
* ``CMAKE_POLICY_WARNING_CMP0065`` controlled the warning for
  policy :policy:`CMP0065` in CMake versions before 4.0.
* ``CMAKE_POLICY_WARNING_CMP0066`` controls the warning for
  policy :policy:`CMP0066`.
* ``CMAKE_POLICY_WARNING_CMP0067`` controls the warning for
  policy :policy:`CMP0067`.
* ``CMAKE_POLICY_WARNING_CMP0082`` controls the warning for
  policy :policy:`CMP0082`.
* ``CMAKE_POLICY_WARNING_CMP0089`` controls the warning for
  policy :policy:`CMP0089`.
* ``CMAKE_POLICY_WARNING_CMP0102`` controls the warning for
  policy :policy:`CMP0102`.
* ``CMAKE_POLICY_WARNING_CMP0112`` controls the warning for
  policy :policy:`CMP0112`.
* ``CMAKE_POLICY_WARNING_CMP0116`` controls the warning for
  policy :policy:`CMP0116`.
* ``CMAKE_POLICY_WARNING_CMP0126`` controls the warning for
  policy :policy:`CMP0126`.
* ``CMAKE_POLICY_WARNING_CMP0128`` controls the warning for
  policy :policy:`CMP0128`.
* ``CMAKE_POLICY_WARNING_CMP0129`` controls the warning for
  policy :policy:`CMP0129`.
* ``CMAKE_POLICY_WARNING_CMP0133`` controls the warning for
  policy :policy:`CMP0133`.
* ``CMAKE_POLICY_WARNING_CMP0172`` controls the warning for
  policy :policy:`CMP0172`.

This variable should not be set by a project in CMake code.  Project
developers running CMake may set this variable in their cache to
enable the warning (e.g. ``-DCMAKE_POLICY_WARNING_CMP<NNNN>=ON``).
Alternatively, running :manual:`cmake(1)` with the
:option:`--debug-output <cmake --debug-output>`,
:option:`--trace <cmake --trace>`, or
:option:`--trace-expand <cmake --trace-expand>` option will also
enable the warning.
