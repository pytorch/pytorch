# <ndk>/build/core/toolchains/arm-linux-androideabi-4.9/setup.mk
string(APPEND _ANDROID_ABI_INIT_CFLAGS
  " -march=armv5te"
  )

if(CMAKE_ANDROID_ARM_MODE)
  string(APPEND _ANDROID_ABI_INIT_CFLAGS " -marm")
else()
  string(APPEND _ANDROID_ABI_INIT_CFLAGS " -mthumb")
endif()

string(APPEND _ANDROID_ABI_INIT_CFLAGS
  " -msoft-float"
  " -mtune=xscale"
  )

include(Platform/Android/abi-common-GNU)
