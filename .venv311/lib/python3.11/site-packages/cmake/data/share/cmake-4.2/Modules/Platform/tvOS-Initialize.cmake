include(Platform/Darwin-Initialize)

if(NOT _CMAKE_OSX_SYSROOT_PATH MATCHES "/AppleTV(OS|Simulator)")
  message(FATAL_ERROR "${CMAKE_OSX_SYSROOT} is not an tvOS SDK")
endif()
