# <ndk>/sources/cxx-stl/llvm-libc++/Android.mk
set(_ANDROID_STL_RTTI 1)
set(_ANDROID_STL_EXCEPTIONS 1)
set(_ANDROID_STL_NOSTDLIBXX 1)
macro(__android_stl_cxx lang filename)
  # Add the include directory.
  if(EXISTS "${CMAKE_ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libcxx/include/cstddef")
    # r12 and below
    __android_stl_inc(${lang} "${CMAKE_ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libcxx/include" 1)
    __android_stl_inc(${lang} "${CMAKE_ANDROID_NDK}/sources/android/support/include" 0)
    __android_stl_inc(${lang} "${CMAKE_ANDROID_NDK}/sources/cxx-stl/llvm-libc++abi/libcxxabi/include" 1)
  else()
    # r13 and above
    __android_stl_inc(${lang} "${CMAKE_ANDROID_NDK}/sources/cxx-stl/llvm-libc++/include" 1)
    __android_stl_inc(${lang} "${CMAKE_ANDROID_NDK}/sources/android/support/include" 0)
    __android_stl_inc(${lang} "${CMAKE_ANDROID_NDK}/sources/cxx-stl/llvm-libc++abi/include" 1)
  endif()

  # Add the library file.
  __android_stl_lib(${lang} "${CMAKE_ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${CMAKE_ANDROID_ARCH_ABI}/${filename}" 1)
endmacro()
