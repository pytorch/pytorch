create_test_sourcelist
----------------------

Create a test driver program that links together many small tests into a
single executable.  This is useful when building static executables with
large libraries to shrink the total required size.

.. signature::
  create_test_sourcelist(<sourceListName> <driverName> <test>... <options>...)
  :target: original

  Generate a test driver source file from a list of individual test sources
  and provide a combined list of sources that can be built as an executable.

  The options are:

  ``<sourceListName>``
    The name of a variable in which to store the list of source files needed
    to build the test driver.  The list will contain the ``<test>...`` sources
    and the generated ``<driverName>`` source.

    .. versionchanged:: 3.29

      The test driver source is listed by absolute path in the build tree.
      Previously it was listed only as ``<driverName>``.

  ``<driverName>``
    Name of the test driver source file to be generated into the build tree.
    The source file will contain a ``main()`` program entry point that
    dispatches to whatever test is named on the command line.

  ``<test>...``
    Test source files to be added to the driver binary.  Each test source
    file must have a function in it that is the same name as the file with the
    extension removed.  For example, a ``foo.cxx`` test source might contain:

    .. code-block:: c++

      int foo(int argc, char** argv)

  ``EXTRA_INCLUDE <header>``
    Specify a header file to ``#include`` in the generated test driver source.

  ``FUNCTION <function>``
    Specify a function to be called with pointers to ``argc`` and ``argv``.
    The function may be provided in the ``EXTRA_INCLUDE`` header:

    .. code-block:: c++

      void function(int* pargc, char*** pargv)

    This can be used to add extra command line processing to each test.

Additionally, some CMake variables affect test driver generation:

.. variable:: CMAKE_TESTDRIVER_BEFORE_TESTMAIN

  Code to be placed directly before calling each test's function.

.. variable:: CMAKE_TESTDRIVER_AFTER_TESTMAIN

  Code to be placed directly after the call to each test's function.
