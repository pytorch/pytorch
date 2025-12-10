include(Platform/Android/ndk-stl-gnustl)
macro(__android_stl lang)
  __android_stl_gnustl(${lang} libgnustl_shared.so)
endmacro()
