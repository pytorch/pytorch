if(__craylinux_crayprgenv_cxx)
  return()
endif()
set(__craylinux_crayprgenv_cxx 1)

include(Compiler/CrayPrgEnv)
__CrayPrgEnv_setup(CXX)
