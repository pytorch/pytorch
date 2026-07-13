# Cross toolchain: x86 host -> aarch64-linux-gnu (for SVE512 cross-compile)
# Usage:
#   cmake -S third_party/sleef -B build-sleef-native -GNinja
#   ninja -C build-sleef-native mkdisp mkrename mkalias
#   cmake -S . -B build-sve512-cross \
#     -DCMAKE_TOOLCHAIN_FILE=<this file> \
#     -DNATIVE_BUILD_DIR=$(pwd)/build-sleef-native \
#     -DUSE_PRIORITIZED_TEXT_FOR_LD=OFF \
#     -GNinja ...
# USE_PRIORITIZED_TEXT_FOR_LD must be OFF: cmake/linker_script.ld is x86_64-only.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_AR aarch64-linux-gnu-ar)
set(CMAKE_RANLIB aarch64-linux-gnu-ranlib)

# Default: Arch aarch64-linux-gnu sysroot (libtorch cross-build).
set(CMAKE_SYSROOT /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)

# Optional: Debian arm64-rootfs path for Python headers/libs only (torch_python).
# Does NOT replace CMAKE_SYSROOT — avoids mixing Debian libc with Arch libstdc++.
set(AARCH64_DEBIAN_ROOTFS "" CACHE PATH "Debian arm64 rootfs for cross Python dev headers/libs")
set(PYTHON_CROSS_INCLUDE_DIR "" CACHE PATH "Minimal include shim for aarch64-linux-gnu/python3.13/pyconfig.h")

if(AARCH64_DEBIAN_ROOTFS)
  list(APPEND CMAKE_FIND_ROOT_PATH "${AARCH64_DEBIAN_ROOTFS}")
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
