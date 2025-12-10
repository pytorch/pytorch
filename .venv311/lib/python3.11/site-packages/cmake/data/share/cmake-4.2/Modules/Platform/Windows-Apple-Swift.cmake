set(CMAKE_Swift_IMPLIB_LINKER_FLAGS "-Xlinker -implib:<TARGET_IMPLIB>")
set(CMAKE_Swift_FLAGS_DEBUG_LINKER_FLAGS "-Xlinker -debug")
set(CMAKE_Swift_FLAGS_RELWITHDEBINFO_LINKER_FLAGS "-Xlinker -debug")

# Linker Selection
set(CMAKE_Swift_USING_LINKER_SYSTEM "-use-ld=link")
set(CMAKE_Swift_USING_LINKER_LLD "-use-ld=lld")
set(CMAKE_Swift_USING_LINKER_MSVC "-use-ld=link")

set(CMAKE_Swift_LINK_MODE DRIVER)
