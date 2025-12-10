This specifies that the current CMake code is written for the given range of
CMake versions, ``<min>[...<max>]``. It sets the "policy version" to:

* the range's ``<max>`` version, if specified, or to
* the ``<min>`` version, or to
* the value of the :variable:`CMAKE_POLICY_VERSION_MINIMUM` variable
  if it is higher than the other two versions.

The policy version effectively requests behavior preferred as of a given CMake
version and tells newer CMake versions to warn about their new policies.
All policies known to the running version of CMake and introduced
in that version or earlier will be set to use ``NEW`` behavior.
All policies introduced in later versions will be unset (unless the
:variable:`CMAKE_POLICY_DEFAULT_CMP<NNNN>` variable sets a default).
This effectively requests behavior preferred as of a given CMake
version and tells newer CMake versions to warn about their new policies.
