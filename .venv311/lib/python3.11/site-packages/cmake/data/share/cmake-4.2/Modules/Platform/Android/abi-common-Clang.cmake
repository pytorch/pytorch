string(APPEND _ANDROID_ABI_INIT_CFLAGS
  #" -Wno-invalid-command-line-argument"
  #" -Wno-unused-command-line-argument"
  )

include(Platform/Android/abi-common)
