include(Platform/Android/ndk-stl-stlport)
macro(__android_stl lang)
  __android_stl_stlport(${lang} libstlport_static.a)
endmacro()
