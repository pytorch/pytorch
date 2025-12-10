# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple linkers; use include blocker.
include_guard()

include(Linker/GNU)

macro(__linker_gnugold lang)
  __linker_gnu(${lang})

  # Due to GNU binutils ld bug when LTO is enabled (see GNU bug
  # `30568 <https://sourceware.org/bugzilla/show_bug.cgi?id=30568>`_),
  # deactivate this feature because all known versions of gold linker have
  # this bug.
  set(CMAKE_${lang}_LINK_DEPENDS_USE_LINKER FALSE)
endmacro()
