# Cross toolchain: x86 host -> aarch64, linking against Debian arm64-rootfs.
# libc/libstdc++ match rootfs python for QEMU user-mode import smoke tests.
#
# Set AARCH64_DEBIAN_ROOTFS via -D or env before cmake (required for try_compile).
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_AR aarch64-linux-gnu-ar)
set(CMAKE_RANLIB aarch64-linux-gnu-ranlib)

if(NOT AARCH64_DEBIAN_ROOTFS)
  if(DEFINED ENV{AARCH64_DEBIAN_ROOTFS})
    set(AARCH64_DEBIAN_ROOTFS "$ENV{AARCH64_DEBIAN_ROOTFS}")
  endif()
endif()
if(NOT AARCH64_DEBIAN_ROOTFS)
  message(FATAL_ERROR "Set -DAARCH64_DEBIAN_ROOTFS=... or export AARCH64_DEBIAN_ROOTFS")
endif()

set(CMAKE_SYSROOT "${AARCH64_DEBIAN_ROOTFS}")
set(CMAKE_FIND_ROOT_PATH "${AARCH64_DEBIAN_ROOTFS}")

set(PYTHON_CROSS_INCLUDE_DIR "" CACHE PATH "Minimal include shim for aarch64-linux-gnu/python3.13/pyconfig.h")

set(_ROOTFS_LIBDIR "${AARCH64_DEBIAN_ROOTFS}/usr/lib/aarch64-linux-gnu")
set(_ROOTFS_LIBDIR2 "${AARCH64_DEBIAN_ROOTFS}/lib/aarch64-linux-gnu")
set(_LINK_FLAGS
  "--sysroot=${AARCH64_DEBIAN_ROOTFS}"
  "-Wl,-rpath-link,${_ROOTFS_LIBDIR}"
  "-Wl,-rpath-link,${_ROOTFS_LIBDIR2}"
  "-L${_ROOTFS_LIBDIR}"
  "-L${_ROOTFS_LIBDIR2}"
)
string(JOIN " " _LINK_FLAGS_STR ${_LINK_FLAGS})
set(CMAKE_EXE_LINKER_FLAGS "${_LINK_FLAGS_STR} ${CMAKE_EXE_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${_LINK_FLAGS_STR} ${CMAKE_SHARED_LINKER_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${_LINK_FLAGS_STR} ${CMAKE_MODULE_LINKER_FLAGS}")

unset(_ROOTFS_LIBDIR)
unset(_ROOTFS_LIBDIR2)
unset(_LINK_FLAGS)
unset(_LINK_FLAGS_STR)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
