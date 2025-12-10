# <ndk>/sources/cxx-stl/stlport/Android.mk
set(_ANDROID_STL_RTTI 1)
set(_ANDROID_STL_EXCEPTIONS 1)
set(_ANDROID_STL_NOSTDLIBXX 0)
macro(__android_stl_stlport lang filename)
  __android_stl_inc(${lang} "${CMAKE_ANDROID_NDK}/sources/cxx-stl/stlport/stlport" 1)
  __android_stl_lib(${lang} "${CMAKE_ANDROID_NDK}/sources/cxx-stl/stlport/libs/${CMAKE_ANDROID_ARCH_ABI}/${filename}" 1)
endmacro()
