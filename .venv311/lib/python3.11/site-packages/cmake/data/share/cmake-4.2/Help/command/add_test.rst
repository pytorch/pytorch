add_test
--------

Add a test to the project to be run by :manual:`ctest(1)`.

.. code-block:: cmake

  add_test(NAME <name> COMMAND <command> [<arg>...]
           [CONFIGURATIONS <config>...]
           [WORKING_DIRECTORY <dir>]
           [COMMAND_EXPAND_LISTS])

Adds a test called ``<name>``.  The test name may contain arbitrary
characters, expressed as a :ref:`Quoted Argument` or :ref:`Bracket Argument`
if necessary.  See policy :policy:`CMP0110`.

CMake only generates tests if the :command:`enable_testing` command has been
invoked.  The :module:`CTest` module invokes ``enable_testing`` automatically
unless :variable:`BUILD_TESTING` is set to ``OFF``.

Tests added with the ``add_test(NAME)`` signature support using
:manual:`generator expressions <cmake-generator-expressions(7)>`
in test properties set by :command:`set_property(TEST)` or
:command:`set_tests_properties`. Test properties may only be set in the
directory the test is created in.

``add_test`` options are:

``COMMAND``
  Specify the test command-line.

  If ``<command>`` specifies an executable target created by
  :command:`add_executable`:

  * It will automatically be replaced by the location of the executable
    created at build time.

  * .. versionadded:: 3.3

      The target's :prop_tgt:`CROSSCOMPILING_EMULATOR`, if set, will be
      used to run the command on the host::

        <emulator> <command>

      .. versionchanged:: 3.29

        The emulator is used only when
        :variable:`cross-compiling <CMAKE_CROSSCOMPILING>`.
        See policy :policy:`CMP0158`.

  * .. versionadded:: 3.29

      The target's :prop_tgt:`TEST_LAUNCHER`, if set, will be
      used to launch the command::

        <launcher> <command>

      If the :prop_tgt:`CROSSCOMPILING_EMULATOR` is also set, both are used::

        <launcher> <emulator> <command>

  The command may be specified using
  :manual:`generator expressions <cmake-generator-expressions(7)>`.

``CONFIGURATIONS``
  Restrict execution of the test only to the named configurations.

``WORKING_DIRECTORY``
  Set the test property :prop_test:`WORKING_DIRECTORY` in which to execute the
  test. If not specified, the test will be run in
  :variable:`CMAKE_CURRENT_BINARY_DIR`. The working directory may be specified
  using :manual:`generator expressions <cmake-generator-expressions(7)>`.

``COMMAND_EXPAND_LISTS``
  .. versionadded:: 3.16

  Lists in ``COMMAND`` arguments will be expanded, including those created with
  :manual:`generator expressions <cmake-generator-expressions(7)>`.

If the test command exits with code ``0`` the test passes. Non-zero exit code
is a "failed" test. The test property :prop_test:`WILL_FAIL` inverts this
logic. Note that system-level test failures such as segmentation faults or
heap errors will still fail the test even if ``WILL_FAIL`` is true. Output
written to stdout or stderr is captured by :manual:`ctest(1)` and only
affects the pass/fail status via the :prop_test:`PASS_REGULAR_EXPRESSION`,
:prop_test:`FAIL_REGULAR_EXPRESSION`, or :prop_test:`SKIP_REGULAR_EXPRESSION`
test properties.

.. versionadded:: 3.16
  Added :prop_test:`SKIP_REGULAR_EXPRESSION` property.

Example usage:

.. code-block:: cmake

  add_test(NAME mytest
           COMMAND testDriver --config $<CONFIG>
                              --exe $<TARGET_FILE:myexe>)

This creates a test ``mytest`` whose command runs a ``testDriver`` tool
passing the configuration name and the full path to the executable
file produced by target ``myexe``.

---------------------------------------------------------------------

The command syntax above is recommended over the older, less flexible form:

.. code-block:: cmake

  add_test(<name> <command> [<arg>...])

Add a test called ``<name>`` with the given command-line.

Unlike the above ``NAME`` signature, target names are not supported
in the command-line.  Furthermore, tests added with this signature do not
support :manual:`generator expressions <cmake-generator-expressions(7)>`
in the command-line or test properties, and the :prop_tgt:`TEST_LAUNCHER`
and :prop_tgt:`CROSSCOMPILING_EMULATOR` target properties are not supported.
