include(Platform/Android/ndk-stl-gabi++)
macro(__android_stl lang)
  __android_stl_gabixx(${lang} libgabi++_static.a)
endmacro()
