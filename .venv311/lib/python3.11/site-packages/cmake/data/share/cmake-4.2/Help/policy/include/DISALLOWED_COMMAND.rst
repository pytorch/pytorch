CMake >= |disallowed_version| prefer that this command never be called.
The ``OLD`` behavior for this policy is to allow the command to be called.
The ``NEW`` behavior for this policy is to issue a ``FATAL_ERROR`` when the
command is called.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: |disallowed_version|
.. |WARNS_OR_DOES_NOT_WARN| replace:: warns
.. include:: include/STANDARD_ADVICE.rst
