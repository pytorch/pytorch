
set(CMAKE_CUDA_VERBOSE_LINK_FLAG "-Wl,-v")

# linker selection
set(CMAKE_CUDA_USING_LINKER_SYSTEM "")
set(CMAKE_CUDA_USING_LINKER_LLD "-fuse-ld=lld")
set(CMAKE_CUDA_USING_LINKER_BFD "-fuse-ld=bfd")
set(CMAKE_CUDA_USING_LINKER_GOLD "-fuse-ld=gold")
set(CMAKE_CUDA_USING_LINKER_MOLD "-fuse-ld=mold")
