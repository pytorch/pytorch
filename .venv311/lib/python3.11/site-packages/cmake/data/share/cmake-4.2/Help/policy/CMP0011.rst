CMP0011
-------

.. |REMOVED_IN_CMAKE_VERSION| replace:: 4.0
.. include:: include/REMOVED_PROLOGUE.rst

Included scripts do automatic :command:`cmake_policy` PUSH and POP.

In CMake 2.6.2 and below, CMake Policy settings in scripts loaded by
the :command:`include` and :command:`find_package` commands would affect
the includer.  Explicit invocations of ``cmake_policy(PUSH)`` and
``cmake_policy(POP)`` were required to isolate policy changes and protect
the includer.  While some scripts intend to affect the policies of their
includer, most do not.  In CMake 2.6.3 and above, :command:`include` and
:command:`find_package` by default ``PUSH`` and ``POP`` an entry on
the policy stack around an included
script, but provide a ``NO_POLICY_SCOPE`` option to disable it.  This
policy determines whether or not to imply ``NO_POLICY_SCOPE`` for
compatibility.  The ``OLD`` behavior for this policy is to imply
``NO_POLICY_SCOPE`` for :command:`include` and :command:`find_package` commands.
The ``NEW`` behavior for this policy is to allow the commands to do
their default cmake_policy ``PUSH`` and ``POP``.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 2.6.3
.. |WARNED_OR_DID_NOT_WARN| replace:: warned
.. include:: include/REMOVED_EPILOGUE.rst
