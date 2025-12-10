# <ndk>/sources/cxx-stl/gabi++/Android.mk
set(_ANDROID_STL_RTTI 1)
set(_ANDROID_STL_EXCEPTIONS 1)
set(_ANDROID_STL_NOSTDLIBXX 1)
macro(__android_stl_gabixx lang filename)
  __android_stl_inc(${lang} "${CMAKE_ANDROID_NDK}/sources/cxx-stl/gabi++/include" 1)
  __android_stl_lib(${lang} "${CMAKE_ANDROID_NDK}/sources/cxx-stl/gabi++/libs/${CMAKE_ANDROID_ARCH_ABI}/${filename}" 1)
endmacro()
