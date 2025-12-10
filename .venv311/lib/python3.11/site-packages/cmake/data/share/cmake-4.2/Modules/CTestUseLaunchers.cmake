# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CTestUseLaunchers
-----------------

This module sets the ``RULE_LAUNCH_*`` global properties when the
:variable:`CTEST_USE_LAUNCHERS` variable is set to a true-like value (e.g.,
``ON``):

* :prop_gbl:`RULE_LAUNCH_COMPILE`
* :prop_gbl:`RULE_LAUNCH_CUSTOM`
* :prop_gbl:`RULE_LAUNCH_LINK`

Load this module in a CMake project with:

.. code-block:: cmake

  include(CTestUseLaunchers)

The ``CTestUseLaunchers`` module is automatically included by the
:module:`CTest` module when ``include(CTest)`` is called.  However, it is
provided as a separate module so that projects can use the
``CTEST_USE_LAUNCHERS`` functionality independently.

To use launchers, set the ``CTEST_USE_LAUNCHERS`` variable to a true-like value
in a :option:`ctest -S` dashboard script, and then also set the
``CTEST_USE_LAUNCHERS`` cache variable in the configured project.  Both
``cmake`` and ``ctest`` must be aware of its value for the launchers to function
correctly:

* ``cmake`` needs it to generate the appropriate build rules
* ``ctest`` requires it for accurate error and warning analysis

For convenience, the environment variable :envvar:`CTEST_USE_LAUNCHERS_DEFAULT`
may be set in the :option:`ctest -S` script.  Then, as long as the
``CMakeLists.txt`` includes the ``CTest`` or ``CTestUseLaunchers`` module, it
will use the value of the environment variable to initialize a
``CTEST_USE_LAUNCHERS`` cache variable.  This cache variable initialization only
occurs if ``CTEST_USE_LAUNCHERS`` is not already defined.

.. versionadded:: 3.8
  If ``CTEST_USE_LAUNCHERS`` is set to a true-like value in a
  :option:`ctest -S` script, the :command:`ctest_configure` command will add
  ``-DCTEST_USE_LAUNCHERS:BOOL=TRUE`` to the ``cmake`` command when configuring
  the project.

Examples
^^^^^^^^

.. code-block:: cmake

  set(CTEST_USE_LAUNCHERS ON)
  include(CTestUseLaunchers)
#]=======================================================================]

if(NOT DEFINED CTEST_USE_LAUNCHERS AND DEFINED ENV{CTEST_USE_LAUNCHERS_DEFAULT})
  set(CTEST_USE_LAUNCHERS "$ENV{CTEST_USE_LAUNCHERS_DEFAULT}"
    CACHE INTERNAL "CTEST_USE_LAUNCHERS initial value from ENV")
endif()

if(NOT "${CMAKE_GENERATOR}" MATCHES "Make|Ninja|FASTBuild")
  set(CTEST_USE_LAUNCHERS 0)
endif()

if(CTEST_USE_LAUNCHERS)
  set(__launch_common_options
    "--target-name <TARGET_NAME> --current-build-dir <CMAKE_CURRENT_BINARY_DIR> --build-dir <CMAKE_BINARY_DIR> --object-dir <TARGET_SUPPORT_DIR>")

  set(__launch_compile_options
    "${__launch_common_options} --output <OBJECT> --source <SOURCE> --language <LANGUAGE>")

  set(__launch_link_options
    "${__launch_common_options} --output <TARGET> --target-type <TARGET_TYPE> --language <LANGUAGE>")

  set(__launch_custom_options
    "${__launch_common_options} --output <OUTPUT>")

  if("${CMAKE_GENERATOR}" MATCHES "Ninja|FASTBuild")
    string(APPEND __launch_compile_options " --filter-prefix <CMAKE_CL_SHOWINCLUDES_PREFIX>")
  endif()

  set(CTEST_LAUNCH_COMPILE
    "\"${CMAKE_CTEST_COMMAND}\" --launch ${__launch_compile_options} --")

  set(CTEST_LAUNCH_LINK
    "\"${CMAKE_CTEST_COMMAND}\" --launch ${__launch_link_options} --")

  set(CTEST_LAUNCH_CUSTOM
    "\"${CMAKE_CTEST_COMMAND}\" --launch ${__launch_custom_options} --")

  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CTEST_LAUNCH_COMPILE}")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CTEST_LAUNCH_LINK}")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_CUSTOM "${CTEST_LAUNCH_CUSTOM}")
endif()
