set(HEADERS
  AlignOf.h
  ArrayRef.h
  Backtrace.h
  C++17.h
  Exception.h
  Flags.h
  flat_hash_map.h
  Half.h
  IdWrapper.h
  intrusive_ptr.h
  LeftRight.h
  logging_is_google_glog.h
  logging_is_not_google_glog.h
  Logging.h
  Metaprogramming.h
  numa.h
  Optional.h
  Registry.h
  SmallVector.h
  string_utils.h
  StringUtil.h
  Type.h
  typeid.h
  TypeList.h
  TypeTraits.h
  UniqueVoidPtr.h
)

set(SOURCES
  Array.cpp
  Backtrace.cpp
  C++17.cpp
  Exception.cpp
  flags_use_gflags.cpp
  flags_use_no_gflags.cpp
  Half.cpp
  intrusive_ptr.cpp
  LeftRight.cpp
  Logging.cpp
  Metaprogramming.cpp
  numa.cpp
  Optional.cpp
  SmallVector.cpp
  StringUtil.cpp
  Type.cpp
  typeid.cpp
  TypeList.cpp
  TypeTraits.cpp
  UniqueVoidPtr.cpp
)

add_library(c10_util STATIC
  ${HEADERS}
  ${SOURCES}
)

c10_setup_visibility(c10_util)

target_include_directories(
    c10_util PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>)

# ---[ Dependency of c10
if (${USE_GFLAGS})
   target_link_libraries(c10_util PUBLIC gflags)
endif()

if (${USE_GLOG})
    target_link_libraries(c10_util PUBLIC glog::glog)
endif()

if (ANDROID)
    target_link_libraries(c10_util PRIVATE log)
endif()

if (USE_NUMA)
  if (NOT CAFFE2_DISABLE_NUMA)
    message(STATUS "NUMA paths:")
    message(STATUS ${Numa_INCLUDE_DIR})
    message(STATUS ${Numa_LIBRARIES})
    include_directories(SYSTEM ${Numa_INCLUDE_DIR})
    target_link_libraries(c10_util PRIVATE ${Numa_LIBRARIES})
  else()
    message(STATUS "NUMA is disabled")
  endif()
else()
  message(STATUS "don't use NUMA")
endif()

install(TARGETS c10_util EXPORT Caffe2Targets DESTINATION lib)
