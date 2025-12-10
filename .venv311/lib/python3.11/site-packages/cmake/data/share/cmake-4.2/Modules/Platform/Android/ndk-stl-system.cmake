# <ndk>/android-ndk-r11c/sources/cxx-stl/system/Android.mk
set(_ANDROID_STL_RTTI 0)
set(_ANDROID_STL_EXCEPTIONS 0)
set(_ANDROID_STL_NOSTDLIBXX 0)
macro(__android_stl lang)
  __android_stl_inc(${lang} "${CMAKE_ANDROID_NDK}/sources/cxx-stl/system/include" 1)
endmacro()
