if (CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  if (NOT USE_OBSERVERS)
    return()
  endif()
else()
  cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
  project(caffe2_observers CXX)
  find_package(Caffe2 REQUIRED)
  option(BUILD_SHARED_LIBS "Build shared libs." ON)
endif()

add_library(caffe2_observers
    "${CMAKE_CURRENT_SOURCE_DIR}/net_observer_reporter_print.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/observer_config.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/perf_observer.cc"
    )
target_link_libraries(caffe2_observers PUBLIC caffe2_library)
target_include_directories(caffe2_observers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
install(TARGETS caffe2_observers DESTINATION lib)
caffe2_interface_library(caffe2_observers caffe2_observers_library)

if (CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  set(Caffe2_MODULES ${Caffe2_MODULES} caffe2_observers_library PARENT_SCOPE)
endif()

