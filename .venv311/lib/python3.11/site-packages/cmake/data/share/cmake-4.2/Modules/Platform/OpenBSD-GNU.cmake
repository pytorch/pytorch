include_guard()

include(Platform/NetBSD-GNU)

macro(__openbsd_compiler_gnu lang)
  __netbsd_compiler_gnu(${lang})
endmacro()
