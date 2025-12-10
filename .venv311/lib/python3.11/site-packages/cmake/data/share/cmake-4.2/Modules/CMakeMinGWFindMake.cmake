# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


find_program(CMAKE_MAKE_PROGRAM mingw32-make.exe PATHS
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\MinGW;InstallLocation]/bin"
  c:/MinGW/bin /MinGW/bin
  "[HKEY_CURRENT_USER\\Software\\CodeBlocks;Path]/MinGW/bin"
  )

mark_as_advanced(CMAKE_MAKE_PROGRAM)
