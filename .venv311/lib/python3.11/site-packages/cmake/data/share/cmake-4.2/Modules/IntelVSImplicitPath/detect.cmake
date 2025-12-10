# look at each path and try to find ifconsol.lib
set(LIB "$ENV{LIB}")
foreach(dir ${LIB})
  file(TO_CMAKE_PATH "${dir}" dir)
  if(EXISTS "${dir}/ifconsol.lib")
    file(WRITE output.cmake "list(APPEND implicit_dirs \"${dir}\")\n")
    break()
  endif()
endforeach()
