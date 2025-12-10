# <ndk>/build/core/toolchains/aarch64-linux-android-4.9/setup.mk

# Suppress -Wl,-z,nocopyreloc flag on arm64-v8a
set(_ANDROID_ABI_INIT_EXE_LDFLAGS_NO_nocopyreloc 1)

include(Platform/Android/abi-common-GNU)
