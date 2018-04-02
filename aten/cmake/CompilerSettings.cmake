if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "--std=c++11 -Wall -Wno-unknown-pragmas -Wno-vla -fexceptions ${CMAKE_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "-fexceptions ${CMAKE_C_FLAGS}")
else()
  # disable some verbose warnings
  set(CMAKE_CXX_FLAGS "/wd4267 /wd4251 /wd4522 /wd4522 /wd4838 /wd4305 /wd4244 /wd4190 /wd4101 /wd4996 /wd4275 ${CMAKE_CXX_FLAGS}")
  # windef.h will define max/min macros if NOMINMAX is not defined
  add_definitions(/DNOMINMAX)
endif()
