CMake >= |disallowed_version| prefer that this command never be called.
The ``OLD`` behavior for this policy is to allow the command to be called.
The ``NEW`` behavior for this policy is to issue a ``FATAL_ERROR`` when the
command is called.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: |disallowed_version|
.. |WARNED_OR_DID_NOT_WARN| replace:: warned
.. include:: include/REMOVED_EPILOGUE.rst
