if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  if(NOT USE_OBSERVERS)
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
if(HAVE_SOVERSION)
  set_target_properties(caffe2_observers PROPERTIES
      VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
endif()
target_link_libraries(caffe2_observers PUBLIC torch_library)
target_include_directories(caffe2_observers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(caffe2_observers PRIVATE "-DCAFFE2_BUILD_OBSERVER_LIB")
install(TARGETS caffe2_observers DESTINATION lib)
caffe2_interface_library(caffe2_observers caffe2_observers_library)
if(MSVC AND BUILD_SHARED_LIBS)
  install(FILES $<TARGET_PDB_FILE:caffe2_observers> DESTINATION lib OPTIONAL)
endif()

if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  set(Caffe2_MODULES ${Caffe2_MODULES} caffe2_observers_library PARENT_SCOPE)
endif()

