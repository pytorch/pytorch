# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This file is used by EnableLanguage in cmGlobalGenerator to
# determine that the selected RC compiler can actually compile
# and link the most basic of programs.   If not, a fatal error
# is set and cmake stops processing commands and will not generate
# any makefiles or projects.

# For now there is no way to do a try compile on just a .rc file
# so just do nothing in here.
set(CMAKE_RC_COMPILER_WORKS 1 CACHE INTERNAL "")
