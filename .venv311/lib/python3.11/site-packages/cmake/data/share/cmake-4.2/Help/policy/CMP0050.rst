CMP0050
-------

.. |REMOVED_IN_CMAKE_VERSION| replace:: 4.0
.. include:: include/REMOVED_PROLOGUE.rst

Disallow add_custom_command SOURCE signatures.

CMake 2.8.12 and lower allowed a signature for :command:`add_custom_command`
which specified an input to a command.  This was undocumented behavior.
Modern use of CMake associates custom commands with their output, rather
than their input.

The ``OLD`` behavior for this policy is to allow the use of
:command:`add_custom_command` SOURCE signatures.  The ``NEW`` behavior for this
policy is to issue an error if such a signature is used.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 3.0
.. |WARNED_OR_DID_NOT_WARN| replace:: warned
.. include:: include/REMOVED_EPILOGUE.rst
