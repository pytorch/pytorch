set(CMAKE_EXE_EXPORTS_Swift_FLAG "-Xclang-linker -Wl,--export-dynamic")

# Linker Selection
# BFD is known to mislink Swift objects resulting in missing type info
set(CMAKE_Swift_USING_LINKER_SYSTEM "")
set(CMAKE_Swift_USING_LINKER_GOLD "-use-ld=gold")
set(CMAKE_Swift_USING_LINKER_LLD "-use-ld=lld")

set(CMAKE_Swift_LINK_MODE DRIVER)
