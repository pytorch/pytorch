CMP0178
-------

.. versionadded:: 3.31

Test command lines preserve empty arguments.

Empty values in the :prop_tgt:`TEST_LAUNCHER` and
:prop_tgt:`CROSSCOMPILING_EMULATOR` target properties are now preserved
for tests added by the following:

* The :command:`add_test` command.
* The :command:`ExternalData_Add_Test` command from the :module:`ExternalData`
  module.
* The :command:`gtest_add_tests` or :command:`gtest_discover_tests` commands
  from the :module:`GoogleTest` module.

For the :command:`gtest_add_tests` and :command:`gtest_discover_tests`
commands, empty elements in the values passed after the ``EXTRA_ARGS``
keyword are also now preserved.

The ``OLD`` behavior of this policy silently discards empty list items
from the :prop_tgt:`TEST_LAUNCHER` and :prop_tgt:`CROSSCOMPILING_EMULATOR`
target properties in the above-mentioned cases.  It also silently discards
empty items from the values given after ``EXTRA_ARGS`` for the
:command:`gtest_add_tests` and :command:`gtest_discover_tests` commands.

The ``NEW`` behavior of this policy preserves empty list items in the
:prop_tgt:`TEST_LAUNCHER` and :prop_tgt:`CROSSCOMPILING_EMULATOR` target
properties, and in values given after ``EXTRA_ARGS`` for
:command:`gtest_add_tests` and :command:`gtest_discover_tests`.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 3.31
.. |WARNS_OR_DOES_NOT_WARN| replace:: warns
.. include:: include/STANDARD_ADVICE.rst

.. include:: include/DEPRECATED.rst
