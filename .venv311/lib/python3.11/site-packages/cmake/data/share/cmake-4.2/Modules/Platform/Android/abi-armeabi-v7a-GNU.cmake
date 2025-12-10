# <ndk>/build/core/toolchains/arm-linux-androideabi-4.9/setup.mk
string(APPEND _ANDROID_ABI_INIT_CFLAGS
  " -march=armv7-a"
  )

if(CMAKE_ANDROID_ARM_MODE)
  string(APPEND _ANDROID_ABI_INIT_CFLAGS " -marm")
else()
  string(APPEND _ANDROID_ABI_INIT_CFLAGS " -mthumb")
endif()

if(CMAKE_ANDROID_ARM_NEON)
  string(APPEND _ANDROID_ABI_INIT_CFLAGS " -mfpu=neon")
else()
  string(APPEND _ANDROID_ABI_INIT_CFLAGS " -mfpu=vfpv3-d16")
endif()

string(APPEND _ANDROID_ABI_INIT_CFLAGS
  " -mfloat-abi=softfp"
  )

string(APPEND _ANDROID_ABI_INIT_LDFLAGS
  " -Wl,--fix-cortex-a8"
  )

include(Platform/Android/abi-common-GNU)
