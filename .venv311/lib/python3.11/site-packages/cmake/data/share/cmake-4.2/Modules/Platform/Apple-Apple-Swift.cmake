set(CMAKE_Swift_SYSROOT_FLAG "-sdk")

# Linker Selections
if("${CMAKE_GENERATOR}" STREQUAL Xcode)
  # Xcode always uses clang to link, regardless of what the cmake link language
  # is. Pass the clang flags when linking with Xcode.
  set(CMAKE_Swift_USING_LINKER_APPLE_CLASSIC "-fuse-ld=ld" "LINKER:-ld_classic")
  set(CMAKE_Swift_USING_LINKER_LLD "-fuse-ld=lld")
  set(CMAKE_Swift_USING_LINKER_SYSTEM "-fuse-ld=ld")
  set(CMAKE_SHARED_MODULE_LOADER_Swift_FLAG "-Wl,-bundle_loader,")
else()
  set(CMAKE_Swift_USING_LINKER_APPLE_CLASSIC "-use-ld=ld" "LINKER:-ld_classic")
  set(CMAKE_Swift_USING_LINKER_LLD "-use-ld=lld")
  set(CMAKE_Swift_USING_LINKER_SYSTEM "-use-ld=ld")
  set(CMAKE_SHARED_MODULE_LOADER_Swift_FLAG "-Xclang-linker -Wl,-bundle_loader,")
  set(CMAKE_SHARED_MODULE_CREATE_Swift_FLAGS "-Xlinker -bundle")
endif()

set(CMAKE_Swift_LINK_MODE DRIVER)
