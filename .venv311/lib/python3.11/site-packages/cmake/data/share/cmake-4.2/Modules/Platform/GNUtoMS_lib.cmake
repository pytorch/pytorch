# Usage: cmake -Dlib=lib.bat -Ddef=out.def -Ddll=out.dll -Dimp=out.dll.a -P GNUtoMS_lib.cmake
get_filename_component(name ${dll} NAME) # .dll file name
string(REGEX REPLACE "\\.dll\\.a$" ".lib" out "${imp}") # .dll.a -> .lib
execute_process(
  COMMAND ${lib} /def:${def} /name:${name} /out:${out}
  RESULT_VARIABLE res
  )
if(res)
  message(FATAL_ERROR "lib failed: ${res}")
endif()
