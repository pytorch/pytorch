include_guard()

include(Platform/OpenBSD-GNU)

macro(__mirbsd_compiler_gnu lang)
  __openbsd_compiler_gnu(${lang})
endmacro()
