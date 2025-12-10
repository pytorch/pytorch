# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CTestScriptMode
---------------

.. note::

  This module is not intended to be included or invoked directly by project
  code.  It is internally used by CTest running in script mode (-S) to
  determine current system.  For usage details refer to the :option:`ctest -S`.
#]=======================================================================]

# Determine the current system, so this information can be used
# in ctest scripts
include(CMakeDetermineSystem)

# Also load the system specific file, which sets up e.g. the search paths.
# This makes the FIND_XXX() calls work much better
include(CMakeSystemSpecificInformation)
