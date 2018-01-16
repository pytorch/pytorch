# requires to be compiled along xttensor

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# C++11
if(CMAKE_VERSION VERSION_LESS "3.1")
  set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

set(src
  APMeter.cc
  AUCMeter.cc
  ClassErrorMeter.cc
  MAPMeter.cc
  MSEMeter.cc
)

add_library(xtmeter ${TH_LINK_STYLE} ${src})
target_link_libraries(xtmeter ATen)

add_executable(test-meter test/basic.cc ${BACKWARD_ENABLE})
# add_backward(test-meter)
target_link_libraries(test-meter xtmeter)
