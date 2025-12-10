# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


find_program(CMAKE_MAKE_PROGRAM make
  REGISTRY_VIEW 32
  PATHS
      # Typical install path for 32-bit MSYS2 (https://repo.msys2.org/distrib/msys2-i686-latest.sfx.exe)
      "C:/msys32/usr"
      # Typical install path for MINGW32 (https://sourceforge.net/projects/mingw)
      "C:/mingw/msys"
      # Git for Windows 32-bit (https://gitforwindows.org/)
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\GitForWindows;InstallPath]/usr")

mark_as_advanced(CMAKE_MAKE_PROGRAM)
