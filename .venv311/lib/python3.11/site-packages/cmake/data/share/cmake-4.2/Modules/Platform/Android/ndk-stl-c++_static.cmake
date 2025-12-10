include(Platform/Android/ndk-stl-c++)
macro(__android_stl lang)
  __android_stl_cxx(${lang} libc++_static.a)
  __android_stl_lib(${lang} "${CMAKE_ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${CMAKE_ANDROID_ARCH_ABI}/libc++abi.a" 0)
  __android_stl_lib(${lang} "${CMAKE_ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${CMAKE_ANDROID_ARCH_ABI}/libandroid_support.a" 0)
  __android_stl_lib(${lang} "${CMAKE_ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${CMAKE_ANDROID_ARCH_ABI}/libunwind.a" 0)
  string(APPEND CMAKE_${lang}_STANDARD_LIBRARIES " -latomic") # provided by toolchain
endmacro()
