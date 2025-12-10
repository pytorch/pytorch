# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


__BlueGeneQ_setup_dynamic(XL CXX)

# -qhalt=s       = Halt on severe error messages
string(APPEND CMAKE_CXX_FLAGS_INIT " -qhalt=s")
